from django.conf.urls.defaults import *

urlpatterns = patterns('',
	(r'^feeds/people/([^/]+)/$', 'osg.views.person'),
	(r'^feeds/people/([^/]+)/friends/$', 'osg.views.friends'),
	(r'^graphs/$', 'osg.views.generate'),
	(r'^graphs/(?P<prog>[a-z]+)/$', 'osg.views.generate'),
	(r'^graphs/(?P<gid>[^/]+)/$', 'osg.views.layout'),
	(r'^graphs/(?P<gid>[^/]+)/(?P<prog>[a-z]+)/$', 'osg.views.layout'),
	(r'^graphs/(?P<gid>[^/]+)/(?P<width>[0-9]+)/(?P<height>[0-9]+)/(?P<prog>[a-z]+)\.(?P<ext>[a-z]+)$', 'osg.views.graphviz'),
)
