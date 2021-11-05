from distutils.core import setup
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d%H%M%S')
    
    return f'dev{date_str}'


__version__ = f'0.0.1.{gen_code()}'


setup(name='ssl_lab',
      version=__version__,
      description='Self-Supervised Lab',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'torch',
            'einops',
            'numpy',
            'timm',
            'ml-collections',
            'jupyterlab',
            'matplotlib',
      ],
      packages=['ssl_lab',
                'ssl_lab.dataset',
                'ssl_lab.network',
                'ssl_lab.network.resnet',
                'ssl_lab.network.vit',
                'ssl_lab.network.byol',
                # trainer
                'ssl_lab.trainer',
                'ssl_lab.trainer.resnet',
                'ssl_lab.trainer.byol',
                'ssl_lab.trainer.vit',
                'ssl_lab.utils',
                'ssl_lab.utils.data'
      ]
)
