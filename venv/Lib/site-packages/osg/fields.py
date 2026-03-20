from django.db.models.fields.related import ReverseManyRelatedObjectsDescriptor
from django.contrib.sites.models import Site
from django.conf import settings
from django.db import models

from string import ascii_letters, digits
from random import choice
import md5

class UniqueIdField(models.CharField):
	"""
	A field derived from CharField which, for convenience, 
	initializes itself to a globally unique value. In the
	generation of this hash, Site is used, and it is assumed
	that the settings file defines SITE_ID.
	"""
	def __init__(self, *av, **kw):
		defaults = {
			'maxlength' : 32,
			'default' : self.uniqueid,
			'unique' : True,
			'editable' : False,
		}
		defaults.update(kw)
		super(UniqueIdField, self).__init__(*av, **defaults)
	def get_internal_type(self):
		return 'CharField'
	def contribute_to_class(self, cls, name):
		super(UniqueIdField, self).contribute_to_class(cls, name)
		self.model = cls
	def uniqueid(self):
		ustr = map(lambda x: choice(ascii_letters + digits),
				range(self.maxlength))
		ustr = ''.join(ustr)
		site = Site.objects.get(pk=settings.SITE_ID)
		uid = md5.new('%s-%s' % (site, ustr)).hexdigest()
		if self.model.objects.filter(**{ self.attname : uid }):
			return self.uniqueid()
		return uid
