#==============================================
#____АЛГОРИТМ_МУЛЬТИ-ОБЪЕКТНОГО_ТРЕКИНГА_______
#____________________SORT______________________
#==============================================

import numpy as np
from filterpy.kalman import KalmanFilter

# Пересечение с объединением
# (intersection over union)
def iou(bb_test, bb_gt):
 
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
  return(o)

# Принимает прямоугольную рамку (bbox) в форме [x1,y1,x2,y2] 
# и возвращает z в форме [x,y,s,r], где:
# x,y - центр рамки, 
# s - масштаб/площадь, 
# r - соотношение сторон
def convert_bbox_to_z(bbox):
  
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

# Принимает прямоугольную рамку в форме [x,y,s,r]
# и возвращает ее в форме [x1,y1,x2,y2], где 
# x1,y1 - левый верхний угол, 
# x2,y2 - правый нижний.
def convert_x_to_bbox(x,score=None):
  
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

# Этот класс представляет внутреннее состояние 
# отдельных отслеживаемых объектов, наблюдаемых как bbox.
class KalmanBoxTracker(object):
  
  count = 0
  # Инициализацият трекера, 
  # с использованием начальной рамки
  def __init__(self,bbox):
    
    # Определение модели постоянной скорости
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. # высокая неопределенность ненаблюдаемым нач. скоростям
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  # Обновление вектора состояний для наблюдаемой bbox
  def update(self,bbox):
    
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  # Продвигает вектор состояния и возвращает 
  # предсказанную оценку граничной области
  def predict(self):
    
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  # Возвращает текущую оценку граничной области
  def get_state(self):
    return convert_x_to_bbox(self.kf.x)

# Присваивает обнаруженные объекты отслеживаемому объекту 
# (оба представлены как bbox)
# Возвращает 3 списка совпадений:
# - несовпадающих обнаружений (unmatched_detections)
# - несовпадающих треков (unmatched_trackers)
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Фильтр совпадений с низким IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  # Установка начальных параметров для сортировки
  def __init__(self, max_age=1, min_hits=3):
    
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  # Метод вызывается один раз для каждого кадра
  # dets - numpy-массив детекций в формате [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
  # Если на кадре ничего не детектируется - передавать [0, 0, 0 ,0 , 0]
  # Возвращает тот же массив, но с дополнительным 6ым элементов - ID объекта
  def update(self, dets=np.empty((0, 5))):
   
    self.frame_count += 1
    # Получаем прогнозируемые местоположения от существующих трекеров
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    # Обновляем сопоставленные трекеры с соответствующими обнаружениями
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # создаем и инициализируем новые трекеры для несовпадающих детекций
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
        i -= 1
        
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


#=======================================

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))