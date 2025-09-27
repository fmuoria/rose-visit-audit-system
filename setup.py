from setuptools import setup, find_packages

setup(
    name='rose-audit-system',
    version='1.0.0',
    description='Automated audit system for ROSE women leaders visit verification',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'geopy>=2.2.0',
        'scikit-learn>=1.1.0',
        'textstat>=0.7.0',
        'openpyxl>=3.0.0'
    ],
    python_requires='>=3.7',
)
