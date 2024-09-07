import numpy as np

def nRMSE(rmse, actual_ic50: np.ndarray):
    return rmse / (max(actual_ic50) - min(actual_ic50))

def correct_ratio(pred_pic50, actual_pic50):
    assert pred_pic50.shape[0] == actual_pic50.shape[0], f"예측값과 실제값의 개수가 다릅니다: {pred_pic50.shape} / {actual_pic50.shape}"
    size = len(pred_pic50)
    return sum(abs(pred_pic50 - actual_pic50) <= .5) / size

def score(rmse, actual_ic50, pred_pic50, actual_pic50):
    return .5 * (1 - min(nRMSE(rmse, actual_ic50), 1)) + .5 * correct_ratio(pred_pic50, actual_pic50)
