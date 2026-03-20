from django.db import models
from django.db.models.fields.related import ReverseManyRelatedObjectsDescriptor
from random import random

class FeedbackDescriptor(ReverseManyRelatedObjectsDescriptor):
	"""
	This is a Django Set Descriptor that implements the generative
	algorithm for feedback networks described by D. R. White,
	N. Kejzar, C. Tsallis, J. D. Farmer, and S. D. White. in
	"A Generative Model for Feedback Networks.", 2006
	http://www.santafe.edu/~jdf/papers/05-08-034.pdf
	"""
	__decay__ = {}
	def __get__(self, instance, instance_type=None):
		manager = super(FeedbackDescriptor, self).__get__(instance, instance_type)
		def _generate(*av, **kw):
			return self.generate(instance, *av, **kw)
		setattr(manager, 'generate', _generate)
		return manager

	def nodes(self, graph):
		attname = self.field.get_attname()
		return getattr(graph, attname)

	def start(self, graph, a):
		"""
		Implementation of eq. 1 for selecting a starting
		node. The argument a, is the parameter alpha.
		"""
		nodes = self.nodes(graph)

		count = nodes.count()

		if count == 0:
			return nodes.create()
		elif count == 1:
			return nodes.get()

		denom = 0.0
		probabilities = []
		for node in nodes.all():
			probability = node.outdegree() ** a
			probabilities.append(probability)
			denom = denom + probability

		selector = random()
		addition = 0.0
		for i in range(count):
			addition = addition + (probabilities[i] / denom)
			if addition > selector:
				break
		return nodes.all()[i]

	def distance(self, b):
		"""
		Implementation of eq. 2 for assignment of search distance.
		The argument b is the parameter beta.
		"""
		if b in self.__decay__:
			denom = self.__decay__[b]
		else:
			print "Calculating denominator for decay", b
			denom = sum(map(lambda x: x ** -b, range(1, 100000)))
			print "Done:", denom
			self.__decay__[b] = denom

		selector = random()
		addition = 0.0
		distance = 0
		while True:
			distance = distance + 1
			addition = addition + (distance ** -b) / denom
			if addition > selector:
				break
		return distance

	def findpath(self, graph, start, distance, g):
		"""
		Implementation of eq. 3 for finding a path to a node
		with which to create a link, subject to the routing	
		paramater g, which corresponds to gamma.
		"""
		def udegree(nodes):
			"""
			utility function to calculate the unused
			degree
			"""
			unused = 0
			for n in nodes:
				if n not in visited:
					unused = unused + 1
			return unused

		print "Starting at", start
		r = start
		visited = [start]

		for step in range(distance):
			neighbours = []
			probabilities = []
			denom = 0.0
			for l in graph.neighbours(r):
				if l in visited:
					continue
				neighbours.append(l)
				probability = 1 + udegree(graph.neighbours(l)) ** g
				probabilities.append(probability)
				denom = denom + probability
			if not neighbours:
				nodes = self.nodes(graph)
				r = nodes.create()
				print "Created", r
				break
			selector = random()
			addition = 0.0
			for i in range(len(neighbours)):
				addition = addition + probabilities[i] / denom
				if addition > selector:
					r = neighbours[i]
					print "Selected", r
					break
		print "Found", r
		print

		return r

	def generate(self, graph, a, b, g, N, *av, **kw):
		"""
		Entry point for graph generation. Repeats the algorithm
		using parameters (a,b,g) = (alpha, beta, gamma) until 
		the network contains N nodes.
		"""
		nodes = self.nodes(graph)
		while nodes.filter(*av, **kw).count() < N:
			start = self.start(graph, a)
			distance = self.distance(b)
			finish = self.findpath(graph, start, distance, g)
			graph.link(start, finish)

class FeedbackField(models.ManyToManyField):
	def contribute_to_class(self, cls, name):
		super(FeedbackField, self).contribute_to_class(cls, name)
		setattr(cls, name, FeedbackDescriptor(self))

class FeedbackGenerator(models.Model):
	__module__ = 'osg.models'
	graphs = models.ManyToManyField('Graph')
	a = models.FloatField(default=0)
	b = models.FloatField(default=2)
	g = models.FloatField(default=0)
	N = models.IntegerField(default=10)

	def generate(self):
		name = '[%s] %s_%s_%s' % (self.id, self.a, self.b, self.g)
		graph, created = self.graphs.get_or_create(name=name)
		if created: self.graphs.add(graph)
		self.regenerate(graph)
		return graph

	def regenerate(self, graph):
		print "regenerating", graph
		graph.nodes.generate(self.a, self.b, self.g, self.N)
		
