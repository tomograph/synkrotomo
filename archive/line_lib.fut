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
  let factor = f32.sqrt(2.0f32)/2.0f32

  -- determine if slope of line rho = x*cost+y*sint returning is less than 0
  let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

  -- find x given y - not designed for vertical lines
  let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-y*sint)/cost

  -- find y given x - not for horizontal lines
  let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          (ray-x*cost)/sint

  let safe_inverse (value: f32) : f32 =
       if value == 0.0 then 0.0 else 1/value

  let inverse (values: []f32) : []f32 =
       map(\v -> safe_inverse v) values

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

   let intersect (plus: f32) (minus: f32) (mini: f32) (maxi: f32): f32=
        -- is zero if both values are below minimum else the positive difference between minus and yplus
        let b = f32.max (plus-mini) 0.0f32
        -- is zero if both values are above maximum else the positive difference between minus and yplus
        let a = f32.max (maxi-minus) 0.0f32
        -- let l = distance left right
        let d = plus-minus
        let minab = f32.min a b
        let u = if minab == 0.0f32 then 0.0f32 else minab/d
        let fact = f32.min u 1
        in fact*f32.sqrt(1.0+d**2.0f32)

   --calculate the intersection lengths between line rho = x*cost+y*sint returning zero if there is no intersection
   let intersectiondistance (sint : f32) (cost: f32) (ray: f32) (lowerleft: point) : f32 =
        let xmin = lowerleft.1
        let xmax = lowerleft.1 + 1.0f32
        let ymin = lowerleft.2
        let ymax = lowerleft.2 + 1.0f32
        let y_left =  find_y xmin ray cost sint
        let x_bottom = find_x ymin ray cost sint
        let x_top = find_x ymax ray cost sint
        let y_right = find_y xmax ray cost sint
        let flat = is_flat cost sint
        let plus = if flat then f32.max y_left y_right else f32.max x_bottom x_top
        let minus = if flat then f32.min y_left y_right else f32.min x_bottom x_top
        let mini = if flat then ymin else xmin
        in intersect plus minus mini (mini+1.0)

   -- get numrhos values starting at rhomin and spaced by deltarho
   let getrhos (rhomin: f32) (deltarho: f32) (numrhos: i32): []f32 =
        map(\s -> rhomin+(r32(s))*deltarho)(iota numrhos)

   -- get minimum rho value of a line on the form rho = x cost + y sint passing through circle with center=center and radius=factor
   let rhomin (cost: f32) (sint: f32) (lowerleft: point) (rhozero: f32) (deltarho: f32): f32 =
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
