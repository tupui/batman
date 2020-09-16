from typing import List, Union, Sequence, Optional, Literal

from pydantic import BaseModel, Field, PositiveInt, PositiveFloat

Array = Sequence[List[float]]


class Sampling(BaseModel):
    method: Literal['halton', 'sobol', 'sobolscramble', 'lhs', 'lhsc',
                    'olhs', 'faure', 'uniform', 'saltelli'] = 'halton'
    init_size: int = Field(20, ge=4)
    distributions: Optional[List[str]]
    discrete: Optional[PositiveInt]


class Resampling(BaseModel):
    method: Literal['discrepancy', 'ego_discrepancy',
                    'sigma_discrepancy', 'sigma', 'loo_sigma', 'loo_sobol',
                    'extrema', 'hybrid', 'optimization'] = 'sigma'
    resamp_size: PositiveInt = 0
    extremum: Literal['min', 'max'] = 'min'
    delta_space: PositiveFloat = 0.08
    q2_criteria: float = Field(0.9, gt=0, lt=1)


class Pod(BaseModel):
    dim_max: PositiveInt = 100
    tolerance: float = Field(0.99, gt=0, lt=1)
    type: Literal['static', 'dynamic']


class Space(BaseModel):
    corners: Array
    sampling: Union[Array, Sampling]
    resampling: Optional[Resampling]


class Function(BaseModel):
    type: Literal['function']
    module: str
    function: str
    discover: Optional[str]


class Coupling(BaseModel):
    coupling_directory: str = 'batman-coupling'
    input_fname: str = 'sample-space.json'
    input_format: str = 'json'
    output_fname: str = 'sample-data.json'
    output_format: str = 'json'


class Hosts(BaseModel):
    hostname: str
    remote_root: str
    username: str
    password: str
    weight: str


class Job(BaseModel):
    type: Literal['job']
    command: str = 'bash script.sh'
    context_directory: str = 'data'
    coupling_directory: str = 'batman-coupling'
    coupling: Coupling
    hosts: Hosts
    clean: bool = False
    discover: Optional[str]


class File(BaseModel):
    type: Literal['file']
    file_pairs: List[str]
    discover: str


class Io(BaseModel):
    space_fname: str = 'sample-space.json'
    space_format: str = 'json'
    data_fname: str = 'sample-data.json'
    data_format: str = 'json'


class Snapthot(BaseModel):
    max_workers: PositiveInt = 1
    plabels: List[str]
    flabels: List[str]
    psizes: Optional[List[int]]
    fsizes: Optional[List[int]]
    provider: Union[Function, Job, File]
    io: Io


class SparceParam(BaseModel):
    max_considered_terms: PositiveInt
    most_significant: PositiveInt
    significance_factor: float
    hyper_factor: float


class Surrogate(BaseModel):
    predictions: Array
    method: Literal['rbf', 'kriging', 'pc', 'evofusion', 'mixture'] = 'kriging'
    multifidelity: bool = False
    cost_ratio: float = Field(2.0, gt=1)
    grand_cost: int = Field(30, ge=4)
    strategy: Literal['Quad', 'LS', 'SparseLS'] = 'Quad'
    degree: int = Field(10, ge=1)
    sparse_param: Optional[SparceParam]
    kernel: Optional[str]
    noise: Union[float, bool] = False
    global_optimizer: bool = True
    clusterer: str = 'cluster.KMeans(n_clusters=2)'
    classifier: str = 'vm.SVC()'
    pca_percentage: float = Field(0.8, gt=0, lt=1)


class UQ(BaseModel):
    sample: int = Field(5000, ge=10)
    test: Optional[str]
    pdf: List[str]
    type: Literal['aggregated', 'block'] = 'aggregated'
    method: Literal['sobol', 'FAST'] = 'sobol'


class Mesh(BaseModel):
    fname: str
    format: str
    xlabel: str
    ylabel: str
    flabels: List[str]
    vmins: List[float]


class Visualization(BaseModel):
    bounds: Array
    doe: bool = True
    resampling: bool = True
    xdata: Array
    axis_disc: List[int]
    flabel: str
    xlabel: str
    plabels: List[str]
    feat_order: List[int]
    ticks_nbr: int = Field(..., ge=4, le=256)
    range_cbar: List[float]
    contours: List[float]
    kiviat_fill: bool = True
    mesh_2D: Mesh


class Settings(BaseModel):
    space: Space
    pod: Optional[Pod]
    snapshot: Snapthot
    surrogate: Optional[Surrogate]
    uq: Optional[UQ]
    visualization: Optional[Visualization]
