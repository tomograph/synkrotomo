module Lines = {
     type point  = ( f32, f32 )

     -- find x given y
     let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if cost == 0 then ray else (ray-y*sint)/cost

     -- find y given x
     let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if sint == 0 then ray else (ray-x*cost)/sint

     let is_flat (cos: f32) (sin: f32): bool =
          sin >= f32.abs(cos)

     let getintersections (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point,point,point) =
          let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint) -- check if y is in grid
          let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval)) -- check if x is in grid
          let p_top = (find_x maxval ray cost sint, maxval) -- check if x is in grid
          let p_right = (maxval, find_y maxval ray cost sint) -- check if y is in grid
          in (p_left,p_bottom,p_top,p_right)

     -- gets entry and exit point in no particular order. might later consider corners and vertical lines on grid edge
     let entryexitPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point) =
          let (p_left,p_bottom,p_top,p_right) = getintersections sint cost ray maxval

          let horizontal = sint == 1
          let vertical = sint == 0
          let flat = is_flat cost sint
          let point1 = if vertical then (ray,-maxval) else if horizontal then (-maxval,ray) else if flat then p_left else p_bottom
          let point2 = if vertical then (ray,maxval) else if horizontal then (maxval,ray) else if flat then p_right else p_top

          in (point1, point2)

     let distance ((x1, y1) : point) ((x2, y2) : point): f32 =
          f32.sqrt( (x2 - x1) ** 2.0f32 + (y2 - y1) ** 2.0f32 )

     let intersectiondistance (sint : f32) (cost: f32) (ray: f32) (pixelcenter: point) : f32 =
          let xmin = pixelcenter.1-0.5f32
          let xmax = pixelcenter.1+0.5f32
          let ymin = pixelcenter.2-0.5f32
          let ymax = pixelcenter.2+0.5f32
          let p_left = (xmin, find_y xmin ray cost sint) -- check if y is in grid
          let p_bottom = (find_x ymin ray cost sint, ymin) -- check if x is in grid
          let p_top = (find_x ymax ray cost sint, ymax) -- check if x is in grid
          let p_right = (xmax, find_y xmax ray cost sint) -- check if y is in grid

          let horizontal = sint == 1
          let vertical = sint == 0

          let point1 = if p_left.2 <= ymax && p_left.2 >= ymin then p_left
                         else if p_bottom.1 <= xmax && p_bottom.1 >= xmin then p_bottom
                         else p_top

          let point2 = if p_right.2 <= ymax && p_right.2 >= ymin then p_right
                         else if p_top.1 <= xmax && p_top.1 >= xmin then p_top
                         else p_bottom

          in (if vertical || horizontal then 1 else distance point1 point2)

     -- convertion to sin/cos arrays of array of radians
     let convert2sincos (angles: []f32) : []point =
          map (\t -> (f32.sin(t),f32.cos(t))) angles

     --convert to entry and exit point
     let convert2entryexit (angles: []f32) (rays: []f32) (maxval: f32): [] (point, point) =
          let sincos = convert2sincos angles
          let anglesrays = flatten(map (\t -> map(\r -> (t.1,t.2,r)) rays) sincos)
          in map(\(s,c,r) -> entryexitPoint s c r maxval) anglesrays

     -- entry point from sin/cos some points may be outside grid!
     let entryPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : point =
          let (p_left,p_bottom,p_top,p_right) = getintersections sint cost ray maxval
          in if f32.abs(p_left.2) <= maxval && sint != 0 then p_left else if p_bottom.1 <= p_top.1 then p_bottom else p_top

     --convert to entry points !!!SHOULD ALSO return cost sint
     let convert2entry (angles: []f32) (rays: []f32) (maxval: f32): [] (point, point) =
          let sincos = convert2sincos angles
          let anglesrays = flatten(map (\t -> map(\r -> (t.1,t.2,r)) rays) sincos)
          in map(\(s,c,r) -> ((entryPoint s c r maxval), (s,c))) anglesrays

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

     let pixelcenter (pix: i32) (size: i32): point=
          let loweryidx = pix/size
          let lowerxidx = pix - size*loweryidx
          let x = lowerxidx - size/2
          let y = loweryidx - size/2
          in (r32(x)+0.5,r32(y)+0.5)

     -- Check whether a point is within the grid.
     let isInGrid (halfsize : f32) (y_step_dir : f32) ((x, y) : point) : bool =
          x >= -1f32*halfsize && x < halfsize && (
               if y_step_dir == -1f32
               then (-1f32*halfsize < y && y <= halfsize)
               else (-1f32*halfsize <= y && y < halfsize)
          )

 }
