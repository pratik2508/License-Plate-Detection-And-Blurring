from celery import Celery
app = Celery('celery_config', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0',
             include=['detect_batch_old', 'detect_batch2'])
app.control.cancel_consumer('sign',destination=['celery@com681'])