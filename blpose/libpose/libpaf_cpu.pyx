# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: cdivision = True
# cython: infer_types = True
# cython: language_level = 3
# distutils: language = c++
# distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

__all__ = ["pad_right_down", "preprocess", "calibrate_output", "find_peaks",
           "pair_points", "pair_limbs", "draw_pose"]

# Python Library
import cv2
import numpy as np
########## Cython Library ##########
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.math cimport sqrt, round, atan2, pi

# need speed test code start
np.import_array()
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
# need speed test code end

# C Type
ctypedef np.uint8_t NP_UINT8
ctypedef np.float32_t NP_FLOAT32

# OpenCV Gaussian Blur Kernel Size
cdef int G_KERNEL_SIZE = 13

# Pose Variable
cdef int _N_KEYPOINTS = 18
cdef int _N_LIMBS = 19
cdef vector[pair[int, int]] _LIMBS_HEATMAP_IDX = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]
cdef vector[pair[int, int]] _LIMBS_PAFMAP_IDX = [
    (28, 29), (12, 13), (14, 15), (16, 17), (20, 21), (22, 23), (24, 25), (0, 1),
    (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (30, 31), (34, 35), (32, 33), (36, 37),
    (18, 19), (26, 27)]
cdef vector[vector[int]] _KEYPOINTS_COLORMAP = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
]

# C Function
cdef pair[vector[float], float] _line_integrate(np.ndarray[NP_FLOAT32, ndim=3] pafmap,
                                                int idx_x, int idx_y, int x_a, int y_a,
                                                int x_b, int y_b, float max_len,
                                                int num_sample):
    cdef int i, mid_x = x_a, mid_y = y_a, vec_x = x_b - x_a, vec_y = y_b - y_a
    cdef float vec_len = sqrt(vec_x * vec_x + vec_y * vec_y)
    cdef float unit_vec_x = vec_x / vec_len, unit_vec_y = vec_y / vec_len
    cdef float sample_x_len = <float> (x_b - x_a) / (num_sample - 1)
    cdef float sample_y_len = <float> (y_b - y_a) / (num_sample - 1)
    cdef vector[float] score_line = vector[float](num_sample)
    cdef float score_with_prior = 0

    for i in range(num_sample - 1):
        mid_x = x_a + <int> round(sample_x_len * i)
        mid_y = y_a + <int> round(sample_y_len * i)
        score_line[i] = pafmap[mid_y, mid_x, idx_x] * unit_vec_x + pafmap[
            mid_y, mid_x, idx_y] * unit_vec_y
        score_with_prior += score_line[i]

    mid_x = x_b
    mid_y = y_b
    score_line[num_sample - 1] = pafmap[mid_y, mid_x, idx_x] * unit_vec_x + pafmap[
        mid_y, mid_x, idx_y] * unit_vec_y
    score_with_prior += score_line[num_sample - 1]

    score_with_prior /= num_sample
    score_with_prior += min(max_len / vec_len - 1, 0)
    return (score_line, score_with_prior)

cdef bool _is_vec_idx_has_n(vector[vector[float]]& vec, int idx, float n):
    cdef int size = vec.size(), i

    if size == 0:
        return False

    for i in range(size):
        if n == vec[i][idx]:
            return True
    return False

cdef vector[vector[float]] _flatten_peaks(vector[vector[vector[float]]]& all_peaks):
    cdef vector[vector[float]] candidate
    cdef int idx_limb, idx_peak, n

    for idx_limb in range(_N_KEYPOINTS):
        n = all_peaks[idx_limb].size()
        for idx_peak in range(n):
            candidate.push_back(all_peaks[idx_limb][idx_peak])
    return candidate

cdef bool _is_vec_has_n(vector[int]& vec, int n):
    cdef int size = vec.size(), i

    if size == 0:
        return False

    for i in range(size):
        if n == vec[i]:
            return True
    return False

cdef bool _is_two_set_can_merge(vector[vector[float]]& all_bodies, int set1, int set2):
    cdef int idx, n_duplicated = 0
    for idx in range(_N_KEYPOINTS):
        if all_bodies[set1][idx] >= 0 and all_bodies[set2][idx] >= 0:
            n_duplicated += 1
    if n_duplicated == 0:
        return True
    else:
        return False

cdef void _rm_vec_ele_by_idx(vector[vector[float]]& all_bodies, vector[int]& idx_del):
    cdef int i, n = idx_del.size()

    for i in range(n):
        all_bodies.erase(all_bodies.begin() + idx_del[i] - i)

cdef float _vec_mean(vector[float]& vec):
    cdef int i, n = vec.size()
    cdef float val = 0

    for i in range(n):
        val += vec[i]
    return val / n

cdef float _rad2deg(float rad):
    return rad * (180 / pi)

# Python Function
def pad_right_down(np.ndarray[NP_UINT8, ndim=3] image, int h, int w, int stride,
                   int pad_value):
    cdef int c = image.shape[2]
    cdef int rem_h = h % stride, rem_w = w % stride
    cdef int pad_h = stride - rem_h if rem_h != 0 else 0, pad_w = stride - rem_w if rem_w != 0 else 0

    cdef np.ndarray[NP_UINT8, ndim=3] padded_image = np.full((h + pad_h, w + pad_w, c),
                                                             pad_value, dtype=np.uint8)

    padded_image[0:h, 0:w, 0:c] = image

    return padded_image

def preprocess(np.ndarray[NP_UINT8, ndim=3] image, int h, int w):
    cdef int c = image.shape[2]
    cdef np.ndarray[NP_FLOAT32, ndim=4] array = np.zeros((1, c, h, w), dtype=np.float32)

    array[0] = np.transpose(image, (2, 0, 1))
    array /= 256.
    array -= 0.5
    return array

def calibrate_output(np.ndarray[NP_FLOAT32, ndim=4] output, int h, int w, int stride):
    cdef int c = output.shape[1]
    cdef np.ndarray[NP_FLOAT32, ndim=3] tmp, calibrated

    tmp = np.transpose(output[0], (1, 2, 0))
    tmp = cv2.resize(tmp, (0, 0), fx=stride, fy=stride)
    calibrated = tmp[0:h, 0:w, 0:c]
    return calibrated

def find_peaks(np.ndarray[NP_FLOAT32, ndim=3] heatmap, int h, int w, float sigma,
               float threshold):
    cdef int c = _N_KEYPOINTS
    cdef int i_c, i_x, i_y, idx = 0
    cdef float tmp, tmp_u, tmp_d, tmp_l, tmp_r
    cdef vector[vector[vector[float]]] all_peaks = vector[vector[vector[float]]](c)

    cdef np.ndarray[NP_FLOAT32, ndim=3] filtered_map = cv2.GaussianBlur(heatmap, (
        G_KERNEL_SIZE, G_KERNEL_SIZE), sigma)

    for i_c in range(c):
        for i_y in range(h):
            for i_x in range(w):
                tmp = filtered_map[i_y, i_x, i_c]
                tmp_u = filtered_map[i_y - 1, i_x, i_c] if i_y - 1 >= 0 else 0
                tmp_d = filtered_map[i_y + 1, i_x, i_c] if i_y + 1 < h else 0
                tmp_l = filtered_map[i_y, i_x - 1, i_c] if i_x - 1 >= 0 else 0
                tmp_r = filtered_map[i_y, i_x + 1, i_c] if i_x + 1 < w else 0
                if tmp >= tmp_u and tmp >= tmp_d and tmp >= tmp_l and tmp >= tmp_r and tmp > threshold:
                    all_peaks[i_c].push_back([i_x, i_y, heatmap[i_y, i_x, i_c], idx])
                    idx += 1
    return all_peaks

def pair_points(np.ndarray[NP_FLOAT32, ndim=3] pafmap,
                vector[vector[vector[float]]]& all_peaks, float max_len, int num_sample,
                float threshold):
    cdef int idx_limb, idx_a, idx_b, idx_x, idx_y
    cdef int n_a, n_b, i_a, i_b, x_a, y_a, x_b, y_b
    cdef vector[vector[vector[float]]] paired_limbs = vector[vector[vector[float]]](
        _N_LIMBS)
    cdef vector[int] unpaired_limb_indices
    cdef vector[vector[float]] limb_cand, connection
    cdef vector[float] scores_line = vector[float](num_sample)
    cdef float score_with_prior
    cdef int n_satisfy, i
    cdef float min_satisfy = 0.8 * num_sample
    cdef int n_cand, i_c, n_connect

    for idx_limb in range(_N_LIMBS):
        idx_a, idx_b = _LIMBS_HEATMAP_IDX[idx_limb]
        idx_x, idx_y = _LIMBS_PAFMAP_IDX[idx_limb]

        n_a = all_peaks[idx_a].size()
        n_b = all_peaks[idx_b].size()
        if n_a != 0 and n_b != 0:
            limb_cand.clear()
            for i_a in range(n_a):
                for i_b in range(n_b):
                    x_a, y_a = all_peaks[idx_a][i_a][0:2]
                    x_b, y_b = all_peaks[idx_b][i_b][0:2]
                    scores_line, score_with_prior = _line_integrate(pafmap, idx_x,
                                                                    idx_y, x_a, y_a,
                                                                    x_b, y_b, max_len,
                                                                    num_sample)
                    n_satisfy = 0
                    for i in range(num_sample):
                        n_satisfy += scores_line[i] > threshold

                    if n_satisfy > min_satisfy and score_with_prior > 0:
                        limb_cand.push_back([i_a, i_b, score_with_prior,
                                             score_with_prior + all_peaks[idx_a][i_a][
                                                 2] + all_peaks[idx_b][i_b][2]])

            connection.clear()
            n_cand = limb_cand.size()
            for i_c in range(n_cand):
                i_a, i_b, score_with_prior = limb_cand[i_c][0:3]
                if not _is_vec_idx_has_n(connection, 3, i_a) and not _is_vec_idx_has_n(
                        connection, 4, i_b):
                    connection.push_back(
                        [all_peaks[idx_a][i_a][3], all_peaks[idx_b][i_b][3],
                         score_with_prior, i_a, i_b])
                    n_connect = connection.size()
                    if n_connect >= min(n_a, n_b):
                        break
            paired_limbs[idx_limb] = connection
        else:
            unpaired_limb_indices.push_back(idx_limb)
    return paired_limbs, unpaired_limb_indices

def pair_limbs(vector[vector[vector[float]]]& all_peaks,
               vector[vector[vector[float]]]& paired_limbs,
               vector[int]& unpaired_limb_indices):
    cdef vector[vector[float]] candidate = _flatten_peaks(all_peaks)
    cdef vector[vector[float]] all_bodies
    cdef vector[float] body
    cdef vector[int] found_set_idx, idx_del
    cdef int idx_limb, idx_a, idx_b, n_paired, idx_paired
    cdef int idx_beg, idx_end, n_found, n_set, idx_set
    cdef float score
    cdef int set1, set2, idx

    for idx_limb in range(_N_LIMBS):
        if not _is_vec_has_n(unpaired_limb_indices, idx_limb):
            idx_a, idx_b = _LIMBS_HEATMAP_IDX[idx_limb]

            n_paired = paired_limbs[idx_limb].size()
            for idx_paired in range(n_paired):
                idx_beg = <int> (paired_limbs[idx_limb][idx_paired][0])
                idx_end = <int> (paired_limbs[idx_limb][idx_paired][1])
                score = paired_limbs[idx_limb][idx_paired][2]
                n_found = 0
                found_set_idx = vector[int](2, -1)
                n_set = all_bodies.size()
                for idx_set in range(n_set):
                    if all_bodies[idx_set][idx_a] == idx_beg or all_bodies[idx_set][
                        idx_b] == idx_end:
                        found_set_idx[n_found] = idx_set
                        n_found += 1
                if n_found == 1:
                    idx_set = found_set_idx[0]
                    if all_bodies[idx_set][idx_b] != idx_end:
                        all_bodies[idx_set][idx_b] = idx_end
                        all_bodies[idx_set][_N_KEYPOINTS + 1] += 1
                        all_bodies[idx_set][_N_KEYPOINTS] += candidate[idx_end][
                                                                 2] + score
                elif n_found == 2:
                    set1, set2 = found_set_idx
                    if _is_two_set_can_merge(all_bodies, set1, set2):
                        for idx in range(_N_KEYPOINTS):
                            all_bodies[set1][idx] += all_bodies[set2][idx] + 1
                        all_bodies[set1][_N_KEYPOINTS] += all_bodies[set2][
                                                              _N_KEYPOINTS] + score
                        all_bodies[set1][_N_KEYPOINTS + 1] += all_bodies[set2][
                            _N_KEYPOINTS + 1]
                        all_bodies.erase(all_bodies.begin() + set2)
                    else:
                        all_bodies[set1][idx_b] = idx_end
                        all_bodies[set1][_N_KEYPOINTS] += candidate[idx_end][2] + score
                        all_bodies[set1][_N_KEYPOINTS + 1] += 1
                elif not n_found and idx_limb < 17:
                    body = vector[float](_N_KEYPOINTS + 2, -1)
                    body[idx_a] = idx_beg
                    body[idx_b] = idx_end
                    body[_N_KEYPOINTS] = candidate[idx_beg][2] + candidate[idx_end][
                        2] + score
                    body[_N_KEYPOINTS + 1] = 2
                    all_bodies.push_back(body)
    n_set = all_bodies.size()
    for idx in range(n_set):
        n_paired = <int> (all_bodies[idx][_N_KEYPOINTS + 1])
        if n_paired < 4 or all_bodies[idx][_N_KEYPOINTS] / n_paired < 0.4:
            idx_del.push_back(idx)
    _rm_vec_ele_by_idx(all_bodies, idx_del)
    return candidate, all_bodies

def draw_pose(np.ndarray[NP_UINT8, ndim=3] image, vector[vector[float]]& candidate,
              vector[vector[float]]& all_bodies):
    cdef int width = 4
    cdef int i_k, i_b, idx, x, y
    cdef int n = all_bodies.size()
    cdef vector[int] idxes = vector[int](2)
    cdef np.ndarray[NP_UINT8, ndim=3] copied
    cdef vector[float] xs = vector[float](2), ys = vector[float](2)
    cdef float x_len, y_len, length, angle

    for i_k in range(_N_KEYPOINTS):
        for i_b in range(n):
            idx = <int> (all_bodies[i_b][i_k])
            if idx == -1:
                continue
            x = <int> (candidate[idx][0])
            y = <int> (candidate[idx][1])
            cv2.circle(image, (x, y), width, _KEYPOINTS_COLORMAP[i_k], thickness=-1)

    for i_k in range(_N_KEYPOINTS - 1):
        for i_b in range(n):
            idxes[0] = <int> (all_bodies[i_b][_LIMBS_HEATMAP_IDX[i_k].first])
            idxes[1] = <int> (all_bodies[i_b][_LIMBS_HEATMAP_IDX[i_k].second])
            if _is_vec_has_n(idxes, -1):
                continue
            copied = image.copy()
            ys[0] = candidate[idxes[0]][0]
            ys[1] = candidate[idxes[1]][0]
            xs[0] = candidate[idxes[0]][1]
            xs[1] = candidate[idxes[1]][1]
            x_len = xs[0] - xs[1]
            y_len = ys[0] - ys[1]
            length = sqrt(x_len * x_len + y_len * y_len)
            angle = _rad2deg(atan2(x_len, y_len))
            cv2.fillConvexPoly(copied, cv2.ellipse2Poly(
                (<int> (_vec_mean(ys)), <int> (_vec_mean(xs))),
                (<int> (length / 2), width), <int> angle, 0, 360, 1),
                               _KEYPOINTS_COLORMAP[i_k])
            image = cv2.addWeighted(image, 0.4, copied, 0.6, 0)
    return image
