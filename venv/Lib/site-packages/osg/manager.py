from django.db import models
from query import GraphQuerySet

class GraphManager(models.Manager):
	def get_query_set(self):
		return GraphQuerySet(model=self.model)

