#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Francisco J. Gonzalez'
SITENAME = 'Francisco J. Gonzalez'
SITEDESCRIPTION = 'This site is '
SITEURL = 'http://localhost:8081'

# plugins
PLUGIN_PATHS = ['../pelican-plugins']
PLUGINS = ['i18n_subsites', 'tipue_search', 'liquid_tags.img', 'render_math']
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}

# theme and theme localization
THEME = '../pelican-fh5co-marble'
TIMEZONE = 'America/Chicago'
DEFAULT_DATE_FORMAT = '%a, %d %b %Y'
DEFAULT_LANG = 'en'
LOCALE = 'en_US'

# content paths
PATH = 'content'
PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['blog']

# static content
STATIC_PATHS = ['images', 'downloads']

# logo path, needs to be stored in PATH Setting
LOGO = '/images/avatar.jpg'

# special content
HERO = [
  {
    'image': '/images/hero/coffee.jpg',
    # for multilanguage support, create a simple dict
    'title': 'Francisco J. Gonzalez',
    'text': 'Ph.D. student @ Illinois',
    'links': []
  },
  # {
  #   'image': '/images/hero/book.jpg',
  #   # keep it a string if you dont need multiple languages
  #   'title': '',
  #   # keep it a string if you dont need multiple languages
  #   'text': '',
  #   'links': [
  #   {
  #     'icon': 'icon-code',
  #     'url': 'https://github.com/claudio-walser/pelican-fh5co-marble',
  #     'text': 'Github'
  #   }
  # ]
  # },
  # {
  #   'image': '/images/hero/background-3.jpg',
  #   'title': 'No Blogroll yet',
  #   'text': 'Because of space issues in the man-nav, i didnt implemented Blogroll links yet.',
  #   'links': []
  # },
  # {
  #   'image': '/images/hero/background-4.jpg',
  #   'title': 'Ads missing as well',
  #   'text': 'And since i hate any ads, this is not implemented as well',
  #   'links': []
  # }
]

# Social widget
SOCIAL = (
  ('LinkedIn', 'https://www.linkedin.com/in/francisco-gonzalez-62a75b9b/'),
  ('Github', 'https://github.com/franjgonzalez'),
)

ABOUT = {
  # 'image': '/images/about/about.jpg',
  'mail': 'fjgonza2@illinois.edu',
  # keep it a string if you dont need multiple languages
  'text': 'Learn more or just drop a message.',
  'link': 'contact.html',
  # the address is also taken for google maps
  'address': '327 Talbot Laboratory  •  104 S. Wright St  •  Urbana, IL 61801',
  'phone': '224-465-6868'
}

# navigation and homepage options
DISPLAY_PAGES_ON_MENU = True
DISPLAY_PAGES_ON_HOME = True
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_TAGS_ON_MENU = False
USE_FOLDER_AS_CATEGORY = True
PAGE_ORDER_BY = 'order'

MENUITEMS = [
  # ('Archive', 'archives.html'),
  ('Contact', 'contact.html')
]

DIRECT_TEMPLATES = [
  'index',
  'tags',
  'categories',
  'authors',
  'archives',
  'search', # needed for tipue_search plugin
  'contact' # needed for the contact form
]

# setup disqus
# DISQUS_SHORTNAME = ''
# DISQUS_ON_PAGES = False # if true its just displayed on every static page, like this you can still enable it per page

# setup google maps
# GOOGLE_MAPS_KEY = 'AIzaSyCefOgb1ZWqYtj7raVSmN4PL2WkTrc-KyA'
