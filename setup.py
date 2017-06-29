from distutils.core import setup
setup(
  name = 'dp4gp',
  packages = ['dp4gp'], # this must be the same as the name above
  version = '1.01',
  description = 'Differential Privacy for Gaussian Processes',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/dp4gp.git',
  download_url = 'https://github.com/lionfish0/dp4gp/archive/1.01.tar.gz',
  keywords = ['differential privacy','gaussian processes'],
  classifiers = [],
  install_requires=['GPy','numpy','sklearn','scipy'],
)
