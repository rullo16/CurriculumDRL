# import torch
# import numpy as np

# @torch.no_grad()
# def distill_reward(camera_obs, student_model=None, teacher_model=None):
#     with torch.no_grad():
#         if not isinstance(camera_obs, torch.Tensor):
#             camera_obs = torch.tensor(camera_obs, dtype=torch.float32)
#         res = (student_model(camera_obs) - teacher_model(camera_obs))
#         res = np.array([res.abs()[0].item() for res in res])
#         return res
    
# class DistillationRewardWrapper():
#     def __init__()