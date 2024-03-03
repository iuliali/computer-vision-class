class Rectangle:
    def __init__(self, left, top, right, bottom):
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        assert left >=0 and right >=0 and top>=0 and bottom>=0
        assert right > left
        assert bottom > top

        self.left   = left          ## x min
        self.top    = top           ## y min
        self.right  = right         ## x max
        self.bottom = bottom        ## y max  
    
    def width(self):
        return self.right - self.left
    
    def height(self):
        return self.bottom - self.top
    
    def top_left(self):
        return (self.left, self.top)
    
    def bottom_right(self):
        return (self.right, self.bottom)
    
    def __str__(self) -> str:
        return f"{self.left}_{self.top}_{self.right}_{self.bottom}"
    
    def intersects(self, other) -> bool:
        if self.left > other.right or self.right < other.left:
            return False
        if self.top > other.bottom or self.bottom < other.top:
            return False
        return True

    def intersects_any(self, others) -> bool:
        for rect in others:
            if self.intersects(rect):
                return True
        return False

    def is_included_in(self, other) -> bool:
        if self.left < other.left or self.right > other.right:
            return False
        if self.top < other.top or self.bottom > other.bottom:
            return False
        return True
