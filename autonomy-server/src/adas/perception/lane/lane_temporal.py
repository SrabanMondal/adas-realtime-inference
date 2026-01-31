import numpy as np

class TemporalLaneTracker:
    def __init__(
        self,
        img_h, img_w,
        alpha=0.85,
        bin_thresh=0.5,
        assign_thresh=25,
        min_points=50
    ):
        self.h = img_h
        self.w = img_w
        self.alpha = alpha
        self.bin_thresh = bin_thresh
        self.assign_thresh = assign_thresh
        self.min_points = min_points

        self.mem_mask = np.zeros((img_h, img_w), dtype=np.float32)

        # Previous polynomials: x = a*y^2 + b*y + c
        self.left_poly = None
        self.right_poly = None

    def ema_update(self, raw_mask):
        self.mem_mask = (
            self.alpha * self.mem_mask +
            (1.0 - self.alpha) * raw_mask
        )
        return self.mem_mask

    def binarize(self):
        return (self.mem_mask > self.bin_thresh).astype(np.uint8)

    def _eval_poly(self, poly, ys):
        # poly: [a, b, c]
        return poly[0] * ys**2 + poly[1] * ys + poly[2]

    def _initial_split(self, xs, ys):
        # Fallback when no history exists
        mid = self.w // 2
        left_idx = xs < mid
        right_idx = xs >= mid
        return xs[left_idx], ys[left_idx], xs[right_idx], ys[right_idx]

    def _temporal_assign(self, xs, ys):
        y = ys.astype(np.float32)

        xl_pred = self._eval_poly(self.left_poly, y)
        xr_pred = self._eval_poly(self.right_poly, y)

        dl = np.abs(xs - xl_pred)
        dr = np.abs(xs - xr_pred)

        left_mask = (dl < dr) & (dl < self.assign_thresh)
        right_mask = (dr < dl) & (dr < self.assign_thresh)

        lx, ly = xs[left_mask], ys[left_mask]
        rx, ry = xs[right_mask], ys[right_mask]

        return lx, ly, rx, ry

    def _fit_poly(self, xs, ys):
        if len(xs) < self.min_points:
            return None
        # Fit x = f(y)
        coeff = np.polyfit(ys, xs, 2)
        return coeff

    def process(self, raw_mask):
        """
        raw_mask: float32 HxW in [0,1]
        Returns:
            left_pts:  [[x,y], ...]
            right_pts: [[x,y], ...]
            left_poly, right_poly
        """
        self.ema_update(raw_mask)
        bin_mask = self.binarize()

        ys, xs = np.where(bin_mask > 0)
        if len(xs) == 0:
            return [], [], self.left_poly, self.right_poly

        if self.left_poly is None or self.right_poly is None:
            lx, ly, rx, ry = self._initial_split(xs, ys)
        else:
            lx, ly, rx, ry = self._temporal_assign(xs, ys)

            # If one side collapses, fallback to spatial split
            if len(lx) < self.min_points or len(rx) < self.min_points:
                lx, ly, rx, ry = self._initial_split(xs, ys)

        new_left_poly = self._fit_poly(lx, ly)
        new_right_poly = self._fit_poly(rx, ry)

        # Keep history if fitting failed
        if new_left_poly is not None:
            self.left_poly = new_left_poly
        if new_right_poly is not None:
            self.right_poly = new_right_poly

        left_pts = np.stack([lx, ly], axis=1).tolist()
        right_pts = np.stack([rx, ry], axis=1).tolist()

        return left_pts, right_pts, self.left_poly, self.right_poly
