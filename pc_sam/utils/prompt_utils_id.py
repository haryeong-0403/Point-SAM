import numpy as np
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN
import sys
import os
from points_visualization import (
    object_points_visualization, 
    visualize_object_clusters_and_centroids, 
    visualized_planes,
    visualize_planes_clusters_and_centroids
    )

import hdbscan

current_file_dir = os.path.dirname(os.path.abspath(__file__))

point_sam_root_dir = os.path.dirname(os.path.dirname(current_file_dir))
sys.path.append(point_sam_root_dir)

def remove_plane(points, dist_threshold=0.06, ransac_n=3, num_iterations=2000):
    """
    RANSAC으로 평면 제거, 평면으로 간주된 나머지 point들은 object로 간주
    """
    Original_points = points.copy()
    wall_planes, floor_planes, ceiling_planes = [],[],[]

    # 최대 30개의 평면을 반복적으로 탐색
    for _ in range(100):
        if len(Original_points) < ransac_n:
            break

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Original_points))
        plane_models, inliers = pcd.segment_plane(dist_threshold, ransac_n, num_iterations)
        # plane_models: [a,b,c,d] -> (ax+by+cz+d=0)
        # inliers: 평면 위에 있다고 판단된 포인트들의 인덱스

        if len(inliers) < 1000:
            # 평면의 크기가 너무 작으면 노이즈로 간주하고 무시
            continue

        plane_pts = Original_points[inliers]

        # RANSAC으로 찾은 평면의 법선 벡터(normal vector)를 정규화하는 과정
        normal = np.array(plane_models[:3]) 
        normal /= np.linalg.norm(normal)

        # 평면 포인트의 중심 좌표 계산 -> 위치 판단용
        center = plane_pts.mean(axis=0)

        # 해당 플래그는: "이번 평면이 우리가 제거할 평면인지?" 여부를 표시하는 변수
        is_plane_used = False
        
        if abs(normal[1]) > 0.9: # 법선이 거의 y축 방향(=수직) -> 바닥 또는 천장 후보
            if center[1] > 0.2: # 중심이 위쪽이면 천장 -> 리스트에 추가하고 제거 대상 표시
                ceiling_planes.append(plane_pts)
                is_plane_used =True

            else: 
                # 중심이 아래쪽이면 바닥
                floor_planes.append(plane_pts)
                is_plane_used = True

        elif abs(normal[0]) > 0.9 or abs(normal[2]) > 0.9:
            # x축 또는 z축 방향이면 -> 수직 평면(벽)
            wall_planes.append(plane_pts)
            is_plane_used = True

        if is_plane_used:
            Original_points = Original_points[np.setdiff1d(np.arange(len(Original_points)), inliers)]
    
    return Original_points, wall_planes, floor_planes, ceiling_planes

def cluster_object_points(points, min_popints=60):
    """
    object 후보 point들을 클러스터링
    """
    if len(points) < min_popints:
        print("prompt_utils_id/cluster_object_points의 point가 부족함")
        return []
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_popints)
    labels = clusterer.fit_predict(points)

    unique_labels = [l for l in set(labels) if l != -1]
    clusters = [points[labels == l] for l in unique_labels]

    return clusters


def sample_centroids(clusters, max_samples=None):
    """
    각 클러스터에서 중심점을 뽑아서 프롬프트로 사용
    """
    centroids = [c.mean(axis=0) for c in clusters if len(c) > 0]
    if max_samples is not None and len(centroids) > max_samples:
        indices = np.random.choice(len(centroids), max_samples, replace=False)
        centroids = [centroids[i] for i in indices]
    
    return centroids

def filter_non_object_points(planes, object_set):
    filtered = []

    for idx, plane in enumerate(planes):
        filtered_plane = []
        removed_cnt = 0

        for pt in plane:
            if tuple(pt) not in object_set:
                filtered_plane.append(pt)
            else:
                removed_cnt += 1

        print(f"[filter] Plane {idx}: 원래 {len(plane)}개 -> 필터링 후 {len(filtered_plane)}개 (제거된 {removed_cnt}개)")

        if filtered_plane:
            filtered.append(np.array(filtered_plane))

    return filtered

def sample_plane_prompts(plane_points, max_samples=300):
    """
    여러 평면으로 구성된 prompt points (list of np.ndarray)에서 
    각 평면을 flatten하고 전체 중 max_samples만 샘플링
    """
    all_points = np.concatenate(plane_points, axis=0)  # (N, 3)

    if len(all_points) <= max_samples:
        return all_points.tolist()
    else:
        sampled_idx = np.random.choice(len(all_points), max_samples, replace=False)
        return all_points[sampled_idx].tolist()


def generate_prompt_points(xyz: np.ndarray, max_object = 10):
    """
    입력 함수 xyz로부터 prompt point, label, instance ID를 자동 생성
    - plane의 모든 point를 background로 사용 (label 0)
    - object의 중심만 foreground (label 순차적으로 번호를 매김)
    """

    object_xyz, wall_planes, floor_planes, ceiling_planes = remove_plane(xyz)

    #####################Visualization############################
    # object라고 판단된 point들 시각화
    # object_points_visualization(object_xyz)
    
    # Planes이라고 판단된 poin들 시각화
    # visualized_planes(wall_planes, floor_planes, ceiling_planes)
    ###############################################################

    # object 클러스터링 및 중심 추출
    object_clusters = cluster_object_points(object_xyz)
    object_prompts = sample_centroids(object_clusters, max_object)

    # object point 집합 저장
    object_set = set(map(tuple, object_xyz))

    # object랑 겹치는 point를 제거한 wall, floor, ceiling
    points_on_walls = filter_non_object_points(wall_planes, object_set)
    points_on_floors = filter_non_object_points(floor_planes, object_set)
    points_on_ceiling = filter_non_object_points(ceiling_planes, object_set)

    # ----------------------filtered wall points--------------------------------------
    # for wall_points in points_on_walls:
    #     print(f"wall points: {wall_points}")
    #     for pt_wall in wall_points:
    #         print(f"pt_wall: {pt_wall}")
    # ---------------------------------------------------------------------------------

    # plane을 이루고 있는 all point를 prompt로 저장
    wall_prompts = []
    ceiling_prompts = []
    floor_prompts = []

    for wall_points in points_on_walls:
        for wall_pt in wall_points:
            wall_prompts.append(wall_pt)

    for ceiling_points in points_on_ceiling:
        for ceiling_pt in ceiling_points:
            ceiling_prompts.append(ceiling_pt)

    for floor_points in points_on_floors:
        for floor_pt in floor_points:
            floor_prompts.append(floor_pt)

    # 전체 prompt + label 구성
    prompt_points = object_prompts + wall_prompts + ceiling_prompts + floor_prompts

    # plane prompt에 label 0 부여
    object_labels  = [1] * len(object_prompts)
    wall_labels    = [0] * len(wall_prompts)
    floor_labels   = [0] * len(floor_prompts)
    ceiling_labels = [0] * len(ceiling_prompts)
    
    # 0/1 구분 label
    prompt_labels = object_labels + wall_labels + floor_labels + ceiling_labels

    # instance ID를 생성하기 위한 영역별 오프셋 값 정의.
    instance_offset = {
        'object': 1000,
    }

    # 각 prompt에 대해 instance ID 부여
    prompt_instances_id = []
    for i in range(len(object_prompts)):
        prompt_instances_id.append(instance_offset['object'] + i)

    prompt_instances_id += [0] * (len(wall_prompts) + len(floor_prompts) + len(ceiling_prompts))
    # print(f"prompt object id: {prompt_instances_id}")

    Np_prompt_points = np.stack(prompt_points)
    Tensor_prompt_objects_id = torch.tensor(prompt_instances_id, dtype=torch.long)
    Tensor_prompt_labels = torch.tensor(prompt_labels, dtype=torch.long)

    return Np_prompt_points, Tensor_prompt_labels, Tensor_prompt_objects_id