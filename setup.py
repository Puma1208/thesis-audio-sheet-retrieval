from setuptools import setup, find_packages

setup(
    name='audio_sheet_retrieval',
    version='0.1',
    packages=find_packages(exclude=['tests', 'tests.*']),
    include_package_data=True,
    url='https://github.com/Puma1208/thesis-audio-sheet-retrieval',
    license='MIT for thesis/ and tests/; audio_sheet_retrieval/ keeps its original terms',
    author='Margarita Naryzhnyaya',
    description='PyTorch audio-to-sheet-music retrieval (MSc thesis), '
                'building on audio_sheet_retrieval by Dorfer et al.',
    python_requires='>=3.10',
)
