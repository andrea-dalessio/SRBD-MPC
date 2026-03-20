from django.utils.translation import ugettext_lazy as _
from django.contrib.gis.db import models
from osg.models import Node

class Location(models.Model):
	__module__ = 'osg.models'
	created = models.DateTimeField(auto_now_add=True,
				verbose_name=_('created'))
	node = models.ForeignKey(Node, related_name='location_set',
				verbose_name=_('node'))
	location = models.PointField(verbose_name=_('location'))
	objects = models.GeoManager()
	class Meta:
		verbose_name=_('location')
		verbose_name_plural=_('locations')
