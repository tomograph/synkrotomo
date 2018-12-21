-- ==
-- compiled input {
--    19
--    0.0f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    27
--    1.57079632679f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    45
--    0.78539816339f32
-- }
-- output {
--    [0.17157287525f32, 0.65685424949f32]
-- }
-- compiled input {
--    36
--    0.78539816339f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    27
--    0.78539816339f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    42
--    2.35619449019f32
-- }
-- output {
--    [0.17157287525f32, 0.65685424949f32]
-- }
-- compiled input {
--    21
--    2.35619449019f32
-- }
-- output {
--    [0.65685424949f32, 0.17157287525f32]
-- }
-- compiled input {
--    18
--    0.78539816339f32
-- }
-- output {
--    [0.65685424949f32, 0.17157287525f32]
-- }

module Lines = {
     type point  = ( f32, f32 )

     -- determine if slope of line rho = x*cost+y*sint returning is less than 0
     let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

     -- find x given y - not designed for vertical lines
     let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-y*sint)/cost

     -- find y given x - not for horizontal lines
     let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-x*cost)/sint

     --calculate the intersection points between line rho = x*cost+y*sint and grid with -maval<=x<=maxval, -maxval <=y<=maxval
     let getintersections (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point,point,point) =
          let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint)
          let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
          let p_top = (find_x maxval ray cost sint, maxval)
          let p_right = (maxval, find_y maxval ray cost sint)
          in (p_left,p_bottom,p_top,p_right)

     -- gets entry and exit point in no particular order. might later consider corners and vertical lines on grid edge
     let entryexitPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point) =
          let flat = is_flat cost sint
          let point1 = if flat then ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint) else (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
          let point2 = if flat then (maxval, find_y maxval ray cost sint) else (find_x maxval ray cost sint, maxval)

          in (point1, point2)

     --calculate the distance bewteen two points
     let distance ((x1, y1) : point) ((x2, y2) : point): f32 =
          f32.sqrt( (x2 - x1) ** 2.0f32 + (y2 - y1) ** 2.0f32 )

     --calculate the intersection lengths between line rho = x*cost+y*sint returning zero if there is no intersection
     let intersectiondistance (sint : f32) (cost: f32) (ray: f32) (lowerleft: point) : f32 =
          let xmin = lowerleft.1
          let xmax = lowerleft.1 + 1.0f32
          let ymin = lowerleft.2
          let ymax = lowerleft.2 + 1.0f32
          let flat = is_flat cost sint
          -- if flat p_left else p_bottom (avoid problems with vertical and horizontal lines this way)
          let ent = if flat then (xmin, find_y xmin ray cost sint) else (find_x ymin ray cost sint, ymin)
          -- if flat p_right else p_top (avoid problems with vertical and horizontal lines this way)
          let ext = if flat then (xmax, find_y xmax ray cost sint) else (find_x ymax ray cost sint, ymax)

          let distance =
          -- if both points outside then distance is zero (they can not be outside on each side of pixel because of the way they get calculated above)
               if (ent.1 < xmin && ext.1 < xmin) || (ent.1 > xmax && ext.1 > xmax) || (ent.2 < ymin && ext.2 < ymin) ||  (ent.2 > ymax && ext.2 > ymax) then 0.0
               else if (ent.1 < xmin && ext.1 > xmax) || (ext.1 < xmin && ent.1 > xmax) || (ent.2 < ymin && ext.2 > ymax)  || (ext.2 < ymin && ent.2 > ymax) then  distance ent ext
          -- if only one point is outside its neither vertical or horizontal and we may safely determine the point in between
          -- if ent.1 < xmin then slope positive and new entry is p_left
               else if ent.1 < xmin then distance (xmin, find_y xmin ray cost sint) ext
          -- if ext.1 < xmin then slope negative and new exit is p_left
               else if ext.1 < xmin then distance (xmin, find_y xmin ray cost sint) ent
          -- same on other side
               else if ent.1 > xmax then distance (xmax, find_y xmax ray cost sint) ext
               else if ext.1 > xmax then distance (xmax, find_y xmax ray cost sint) ent
          -- same for y points outside
               else if ent.2 < ymin then distance (find_x ymin ray cost sint, ymin) ext
          -- if ext.1 < xmin then slope negative and new exit is p_left
               else if ext.2 < ymin then distance (find_x ymin ray cost sint, ymin) ent
          -- same on other side
               else if ent.2 > ymax then distance (find_x ymax ray cost sint, ymax) ext
               else if ext.2 > ymax then distance (find_x ymax ray cost sint, ymax) ent
               else distance ent ext

          in distance

     -- get numrhos values starting at rhomin and spaced by deltarho
     let getrhos (rhomin: f32) (deltarho: f32) (numrhos: i32): []f32 =
          map(\s -> rhomin+(r32(s))*deltarho)(iota numrhos)

     -- get minimum rho value of a line on the form rho = x cost + y sint passing through circle with center=center and radius=factor
     let rhomin (cost: f32) (sint: f32) (lowerleft: point) (rhozero: f32) (deltarho: f32): f32 =
          let factor = f32.sqrt(2.0f32)/2.0f32
          let p = (lowerleft.1+0.5f32-factor*cost, lowerleft.2+0.5f32-factor*sint)
          let rho = cost*p.1+sint*p.2
          let s = f32.ceil((rho-rhozero)/deltarho)
          in rhozero+s*deltarho

     -- get the center coordinate of a pixel
     let lowerleftpixelpoint (pix: i32) (size: i32): point=
          let loweryidx = pix/size
          let lowerxidx = pix - size*loweryidx
          let x = lowerxidx - size/2
          let y = loweryidx - size/2
          in (r32(x),r32(y))
 }

 open Lines

 let main  (pixel : i32)
           (angle: f32): [2]f32 =
           let lowerleft = lowerleftpixelpoint pixel 8
           let cost = f32.cos(angle)
           let sint = f32.sin(angle)
           let min = rhomin cost sint lowerleft (-5.5) 1
           let rhos = getrhos min 1 2
           in map(\r -> intersectiondistance sint cost r lowerleft)rhos
