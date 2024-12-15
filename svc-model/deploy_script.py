import sagemaker
from sagemaker.sklearn import SKLearn

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='model/train.py',
    role='arn:aws:iam::430238166084:role/chetan_oidc_aws_github',
    max_run=60,
    instance_count=1,
    instance_type='ml.t3.large',
    use_spot_instances=True,
    max_wait=120,
    max_retry_attempts=1,
    framework_version='0.23-1',
    py_version='py3',    
    hyperparameters={
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    }
)

# Train the model
sklearn_estimator.fit({
    #'training': 's3://your-bucket/path/to/training/data'
    'training': 's3://ml-iris-chetan/'
})

# Deploy the model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t3.large'
)