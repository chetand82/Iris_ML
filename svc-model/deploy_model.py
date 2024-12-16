import sagemaker
from sagemaker.model import Model

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::430238166084:role/chetan_oidc_aws_github'

model = Model(
    image_uri='430238166084.dkr.ecr.us-east-1.amazonaws.com/svc-sagemaker-model:latest',
    #image_uri='<your-account-id>.dkr.ecr.<region>.amazonaws.com/svc-sagemaker-model:latest',
    entry_point='model/train.py',
    model_data=None,  # No separate model artifact
    role='arn:aws:iam::430238166084:role/chetan_oidc_aws_github',    
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    #instance_type='ml.m5.xlarge',
    initial_instance_count=1,
    max_run=60,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    use_spot_instances=True,
    max_wait=120,
    max_retry_attempts=1,
    py_version='py3',    
    hyperparameters={
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    }
)