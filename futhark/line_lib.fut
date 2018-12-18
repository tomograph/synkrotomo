-- ==
-- compiled input {
--    1
--    0.0f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    5
--    1.57079632679f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    15
--    0.78539816339f32
-- }
-- output {
--    [0.17157287525f32, 0.65685424949f32]
-- }
-- compiled input {
--    10
--    0.78539816339f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    5
--    0.78539816339f32
-- }
-- output {
--    [1.0f32, 0.0f32]
-- }
-- compiled input {
--    12
--    2.35619449019f32
-- }
-- output {
--    [0.17157287525f32, 0.65685424949f32]
-- }
-- compiled input {
--    3
--    2.35619449019f32
-- }
-- output {
--    [0.65685424949f32, 0.17157287525f32]
-- }
-- compiled input {
--    0
--    0.78539816339f32
-- }
-- output {
--    [0.65685424949f32, 0.17157287525f32]
-- }

module Lines = {
     type point  = ( f32, f32 )
     -- error margin
     let stddev = 0.000000005f32

     -- determine if slope of line rho = x*cost+y*sint returning is less than 0
     let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

     let is_vertical (cos: f32): bool =
          f32.abs(cos) >= 1.0-stddev

     let is_horizontal (sin: f32): bool =
          f32.abs(sin) >= 1.0-stddev

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
     let intersectiondistance (sint : f32) (cost: f32) (ray: f32) (pixelcenter: point) : f32 =
          let xmin = pixelcenter.1-0.5f32
          let xmax = pixelcenter.1+0.5f32
          let ymin = pixelcenter.2-0.5f32
          let ymax = pixelcenter.2+0.5f32
          let p_left = (xmin, find_y xmin ray cost sint)
          let p_bottom = (find_x ymin ray cost sint, ymin)
          let p_top = (find_x ymax ray cost sint, ymax)
          let p_right = (xmax, find_y xmax ray cost sint)

          let point1 = if p_left.2 <= ymax && p_left.2 >= ymin then p_left
                         else if p_bottom.1 <= xmax && p_bottom.1 >= xmin then p_bottom
                         else if p_top.1 <= xmax && p_top.1 >= xmin then p_top
                         else if p_right.2 <= ymax && p_right.2 >= ymin then p_right
                         else pixelcenter

          let point2 = if p_right.2 <= ymax && p_right.2 >= ymin then p_right
                         else if p_top.1 <= xmax && p_top.1 >= xmin then p_top
                         else if p_bottom.1 <= xmax && p_bottom.1 >= xmin then p_bottom
                         else if p_left.2 <= ymax && p_left.2 >= ymin then p_left
                         else pixelcenter

          in distance point1 point2

     -- get numrhos values starting at rhomin and spaced by deltarho
     let getrhos (rhomin: f32) (deltarho: f32) (numrhos: i32): []f32 =
          let rhomins = replicate numrhos rhomin
          let iot = map(\r -> r32(r))(iota numrhos)
          let deltas = replicate numrhos deltarho
          in map2 (+) rhomins (map2 (*) iot deltas)

     -- get minimum rho value of a line on the form rho = x cost + y sint passing through circle with center=center and radius=factor
     let rhomin (cost: f32) (sint: f32) (center: point) (rhozero: f32) (deltarho: f32): f32 =
          let factor = f32.sqrt(2)/2
          let p1 = (center.1+factor*cost, center.2+factor*sint)
          let p2 = (center.1-factor*cost, center.2-factor*sint)
          let rho1 = cost*p1.1+sint*p1.2
          let rho2 = cost*p2.1+sint*p2.2
          let rhominfloat = if rho1 < rho2 then rho1 else rho2
          let s = f32.ceil((rhominfloat-rhozero)/deltarho)
          in rhozero+s*deltarho

     -- get the center coordinate of a pixel
     let pixelcenter (pix: i32) (size: i32): point=
          let loweryidx = pix/size
          let lowerxidx = pix - size*loweryidx
          let x = lowerxidx - size/2
          let y = loweryidx - size/2
          in (r32(x)+0.5,r32(y)+0.5)
 }

 open Lines

 let main  (pixel : i32)
           (angle: f32): [2]f32 =
           let center = pixelcenter pixel 4
           let cost = f32.cos(angle)
           let sint = f32.sin(angle)
           let min = rhomin cost sint center (-3.5) 1
           let rhos = getrhos min 1 2
           in map(\r -> intersectiondistance sint cost r center)rhos
