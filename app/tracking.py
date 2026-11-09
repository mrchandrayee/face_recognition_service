import math

class CentroidTracker:
    """Lightweight centroid tracker for short-lived IDs within one video."""
    def __init__(self, max_disappeared=20, max_distance=90):
        self.next_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, boxes):
        centers = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes]
        assigned, used = {}, set()
        for i, center in enumerate(centers):
            best_id, best_dist = None, self.max_distance
            for oid, old in self.objects.items():
                if oid in used: continue
                dist = math.dist(center, old)
                if dist < best_dist:
                    best_id, best_dist = oid, dist
            if best_id is not None:
                assigned[i] = best_id
                used.add(best_id)
                self.objects[best_id] = center
                self.disappeared[best_id] = 0
            else:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = center
                self.disappeared[oid] = 0
                assigned[i] = oid
        for oid in list(self.objects):
            if oid not in used and oid not in assigned.values():
                self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
        return assigned
