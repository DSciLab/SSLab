from distutils.core import setup
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d%H%M%S')
    
    return f'dev{date_str}'


__version__ = f'0.0.1.{gen_code()}'


setup(name='trans_lab',
      version=__version__,
      description='Transformer Lab',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'torch',
            'einops',
            'yacs',
            'easydict',
            'numpy',
            'timm',
            'ml-collections',
            'jupyterlab',
            'matplotlib',
      ],
      packages=['trans_lab',
                'trans_lab.dataset',
                'trans_lab.network',
                'trans_lab.network.resnet',
                'trans_lab.network.mobile',
                'trans_lab.network.fgvc',
                'trans_lab.network.wsdan',
                'trans_lab.network.trans_fg',
                'trans_lab.network.vit',
                'trans_lab.network.tnt',
                'trans_lab.network.sef',
                'trans_lab.network.dpt',
                'trans_lab.network.dpt.dpt',
                'trans_lab.network.dpt.dpt.msda',
                'trans_lab.network.vit_ps',
                'trans_lab.network.vit_ps.layers',
                'trans_lab.network.tfg',
                'trans_lab.network.cait',
                'trans_lab.network.cvt',
                'trans_lab.network.t2t',
                'trans_lab.network.pit',
                'trans_lab.network.cct',
                'trans_lab.network.cct.utils',
                'trans_lab.network.t2t_official',
                'trans_lab.network.cross_vit',
                'trans_lab.network.mix_tfg',
                'trans_lab.network.deep_vit',
                'trans_lab.network.le_vit',
                'trans_lab.network.nest',
                'trans_lab.network.twins_svt',
                'trans_lab.network.iRPE',
                'trans_lab.network.iRPE.rpe_ops',
                # trainers
                'trans_lab.trainer',
                'trans_lab.trainer.resnet',
                'trans_lab.trainer.mobile',
                'trans_lab.trainer.fgvc',
                'trans_lab.trainer.wsdan',
                'trans_lab.trainer.trans_fg',
                'trans_lab.trainer.trans_fg.utils',
                'trans_lab.trainer.vit',
                'trans_lab.trainer.tnt',
                'trans_lab.trainer.sef',
                'trans_lab.trainer.dpt',
                'trans_lab.trainer.cait',
                'trans_lab.trainer.pvt',
                'trans_lab.trainer.vit_ps',
                'trans_lab.trainer.tfg',
                'trans_lab.trainer.t2t',
                'trans_lab.trainer.pit',
                'trans_lab.trainer.cvt',
                'trans_lab.trainer.le_vit',
                'trans_lab.trainer.deep_vit',
                'trans_lab.trainer.t2t_official',
                'trans_lab.trainer.cross_vit',
                'trans_lab.trainer.tfg.utils',
                'trans_lab.trainer.cct',
                'trans_lab.trainer.nest',
                'trans_lab.trainer.iRPE',
                'trans_lab.trainer.twins_svt',
                'trans_lab.trainer.mix_tfg',
                'trans_lab.trainer.mix_tfg.utils',
                'trans_lab.utils',
                'trans_lab.utils.data'
      ]
)
