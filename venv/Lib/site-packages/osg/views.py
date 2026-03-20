from django.http import Http404, HttpResponse, HttpResponseRedirect
from django.utils.encoding import force_unicode, iri_to_uri
from django.utils.translation import ugettext as _
from django.utils.xmlutils import SimplerXMLGenerator
from django.template import loader, RequestContext
from django.shortcuts import get_object_or_404
from django.contrib.sites.models import Site
from django import newforms as forms
from django.conf import settings
from datetime import datetime

from networkx import to_agraph
from osg.models import Node, Graph, FeedbackGenerator

namespaces = {
	'xmlns' : 'http://www.w3.org/2005/Atom',
	'xmlns:georss' : 'http://www.georss.org/georss',
	'xmlns:gd' : 'http://schemas.google.com/g/2005',
}

def get_node_or_404(guid):
	try: 
		return Node.objects.get(uniqueid = guid)
	except Node.DoesNotExist:
		raise Http404

def node_uri(node):
	site = Site.objects.get(pk=settings.SITE_ID)
	return 'http://%s/feeds/people/%s/' % (site.domain, node.uniqueid)

def node_name(node):
	names = map(str, node.name_set.all())
	return '\n'.join(names)

def person(request, guid):
	"""
	Support for OpenSocial Person API 
	http://code.google.com/apis/opensocial/docs/gdata/people/developers_guide_protocol.html#GettingPerson
	"""
	node = get_node_or_404(guid)
	response = HttpResponse(mimetype='text/xml')
	handler = SimplerXMLGenerator(response, 'utf-8')
	entry(handler, node, namespaces)
	return response

def friends(request, guid):
	"""
	Support for OpenSocial Friends API
	http://code.google.com/apis/opensocial/docs/gdata/people/developers_guide_protocol.html#RetrievingFriends
	"""
	node = get_node_or_404(guid)
	response = HttpResponse(mimetype='text/xml')
	handler = SimplerXMLGenerator(response, 'utf-8')

	handler.startElement(u'feed', namespaces)

	uri = node_uri(node)
	handler.addQuickElement(u'id', uri + 'friends/')

	updated = datetime.utcnow().isoformat() + '-00:00'
	handler.addQuickElement(u'updated', updated)

	handler.addQuickElement(u'title', _('friends'))

	name = node_name(node)
	if name:
		handler.startElement(u'author', {})
		handler.addQuickElement(u'name', name)
		handler.endElement(u'author')

	for neighbour in node.neighbours():
		entry(handler, neighbour)

	handler.endElement(u'feed')

	return response

def entry(handler, node, namespaces = {}):
	handler.startElement(u'entry', namespaces)

	uri = node_uri(node)
	handler.addQuickElement(u'id', uri)

	updated = datetime.utcnow().isoformat() + '-00:00'
	handler.addQuickElement(u'updated', updated)

	title = node_name(node)
	if title:
		handler.addQuickElement(u'title', title)

	link_self_attrs = {
		'rel' : 'self',
		'type' : 'application/atom+xml',
		'href' : uri,
	}
	handler.addQuickElement(u'link', u'', link_self_attrs)

	handler.endElement(u'entry')

###
### graph visualization, depends on networkx
###
from os import path, stat, makedirs, unlink, fdopen, system
from tempfile import mkstemp

def render_cachefile(obj, filename, callback, flush=False):
	if hasattr(settings, 'RENDER_CACHE'):
		cachedir = settings.RENDER_CACHE
	else:
		cachedir = './var/neato'
	cachedir = cachedir + '%s/%s/' % (obj._meta.object_name, obj.id)
	subpath, basename = path.split(filename)
	cachedir = path.join(cachedir, subpath)
	try: stat(cachedir)
	except OSError: makedirs(cachedir)

	cachefile = path.join(cachedir, basename)
	try:
		exists = stat(cachefile)
	except OSError:
		exists = False
	if flush and exists:
		unlink(cachefile)
	if flush or not exists:
		callback(obj, cachefile)
	return cachefile

def render_gv(graph, filename, prog, ext, **kw):
	graph_attr = { 'graph':{}, 'node':{}, 'edge':{} }
	graph_attr['graph']['label'] = str(graph)

	node_attr = {}
	edge_attr = {}
	g = graph.graph()

	fd, temp = mkstemp('.' + ext)
	fp = fdopen(fd, 'r')
	fp.close()

	print "rendering", temp
	a = to_agraph(g,
		graph_attr=graph_attr,
		node_attr=node_attr,
		edge_attr=edge_attr)
	a.layout(prog=prog)
	a.draw(temp)
	print "finished"

	print "converting", filename
	cmd = "convert %s -resize %dx%d %s" % (
		temp,
		kw['width'], kw['height'],
		filename,
	)
	print cmd
	system(cmd)
	print "finished"

def get_graph_or_404(gid):
	try:
		return Graph.objects.get(id=int(gid))
	except Graph.DoesNotExist:
		raise Http404

def graphviz(request, gid, prog='neato', ext='png', flush=False, **kw):
	graph = get_graph_or_404(gid)
	filename = '.'.join((prog, ext))
	if 'flush' in request.GET:
		flush=True
	else:
		flush=False

	for key, default in (
			('height', 600),
			('width', 600),
	):
		kw[key] = int(kw.get(key, default))
		filename = path.join(str(kw[key]), filename)

	def _render(graph, filename):
		return render_gv(graph, filename, prog, ext, **kw)
	cachefile = render_cachefile(graph, filename, _render, flush=flush)
	fp = open(cachefile, 'r')
	data = fp.read()
	response = HttpResponse(data, mimetype='image/png')
	fp.close()
	return response

def layout(request, gid, prog='neato', template='osg/graph_layout.html'):
	graph = get_graph_or_404(gid)
	next = Graph.objects.filter(created__gt=graph.created).order_by("created")[:3]
	recent = Graph.objects.filter(created__lt=graph.created).order_by("-created")
	recent = recent[:6-max(len(next),3)]
	next = list(next)
	recent = list(recent)
	next.reverse()
	graphs = next + recent
	context = {
		'graphs': graphs,
		'graph' : graph,
		'info' : graph.graph().info(),
		'prog' : prog,
		'layouts' : ('neato', 'twopi', 'fdp', 'dot'),
	}
	t = loader.get_template(template)
	c = RequestContext(request, context)
	return HttpResponse(t.render(c))

class GeneratorForm(forms.Form):
	a = forms.FloatField(initial=0.0)
	b = forms.FloatField(initial=2.0)
	g = forms.FloatField(initial=0.0)
	N = forms.FloatField(initial=5)
	name = forms.CharField()

def generate(request, prog='neato', template='osg/graph_generate.html'):
	if request.POST:
		f = GeneratorForm(request.POST)
		if f.is_valid() and request.user.is_admin:
			kw = {}
			for k in ('a', 'b', 'g'):
				kw[k] = f.cleaned_data[k]
			g, created = FeedbackGenerator.objects.get_or_create(**kw)
			g.N = f.cleaned_data['N']
			graph = g.generate()
			graph.name = '%s %s' % (graph.name, f.cleaned_data['name'])
			graph.save()
			return HttpResponseRedirect('/graphs/%s/%s/' % (graph.id, prog))
	else:
		f = GeneratorForm()
	context = {
		'form' : f,
		'prog' : prog,
		'graphs' : Graph.objects.all()[:6],
	}
	t = loader.get_template(template)
	c = RequestContext(request, context)
	return HttpResponse(t.render(c))
