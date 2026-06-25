打开Redis && Milvus
启动程序: uvicorn api:app --reload
启动celery: celery -A app.Redis_Celery.celery_app:celery_app worker \
  --loglevel=info \
  --pool=solo
运行benchmark
cd /Users/eric_zcz/Desktop/Eric_Project/agent/Project/week4
python3 tests/benchmark_api.py --endpoint local_query --counts 1 5 10 --mode serial


