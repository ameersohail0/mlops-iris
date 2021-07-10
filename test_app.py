from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["ping"] == "pong" # Modifying structure of test, to accomodate timestamp


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Virginica" # Modifying structure of test, to accomodate timestamp

# Task 2
# Added 2 Test Cases for Iris Versicolour and Iris Setosa

# test to check if Iris Versicolour is classified correctly
def test_pred_versicolour():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 4.9,
        "sepal_width": 2.4,
        "petal_length": 3.3,
        "petal_width": 1.0,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Versicolour" # Modifying structure of test, to accomodate timestamp

# test to check if Iris Setosa is classified correctly
def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.2,
        "sepal_width": 4.1,
        "petal_length": 1.5,
        "petal_width": 0.1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Setosa" # Modifying structure of test, to accomodate timestamp