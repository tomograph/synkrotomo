module Lines = {
     type point  = ( f32, f32 )
     -- error margin
     let stddev = 0.0000005f32

     -- find x given y
     let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if f32.abs(cost)-stddev <= 0 then ray else (ray-y*sint)/cost

     -- find y given x
     let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if f32.abs(sint)-stddev <= 0 then ray else (ray-x*cost)/sint

     -- determine if slope of line rho = x*cost+y*sint returning is less than 0
     let is_flat (cos: f32) (sin: f32): bool =
          f32.abs(sin) >= f32.abs(cos)

     let is_vertical (cos: f32): bool =
          f32.abs(cos) - 1 + stddev >= 0

     let is_horizontal (sin: f32): bool =
          f32.abs(sin) - 1 + stddev >= 0

     --calculate the intersection points between line rho = x*cost+y*sint and grid with -maval<=x<=maxval, -maxval <=y<=maxval
     let getintersections (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point,point,point) =
          let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint)
          let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
          let p_top = (find_x maxval ray cost sint, maxval)
          let p_right = (maxval, find_y maxval ray cost sint)
          in (p_left,p_bottom,p_top,p_right)

     -- gets entry and exit point in no particular order. might later consider corners and vertical lines on grid edge
     let entryexitPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point) =
          let (p_left,p_bottom,p_top,p_right) = getintersections sint cost ray maxval

          let horizontal = is_horizontal(sint)
          let vertical = is_vertical(cost)

          let flat = is_flat cost sint
          let point1 = if vertical then (ray,-maxval) else if horizontal then (-maxval,ray) else if flat then p_left else p_bottom
          let point2 = if vertical then (ray,maxval) else if horizontal then (maxval,ray) else if flat then p_right else p_top

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

          let horizontal = is_horizontal(sint)
          let vertical = is_vertical(cost)

          let point1 = if horizontal && ray >= ymin && ray <= ymax then (xmin, ray)
                         else if vertical && ray >= xmin && ray <= xmax then (ray,ymin)
                         else if vertical || horizontal then pixelcenter
                         else if p_left.2 <= ymax && p_left.2 >= ymin then p_left
                         else if p_bottom.1 <= xmax && p_bottom.1 >= xmin then p_bottom
                         else if p_top.1 <= xmax && p_top.1 >= xmin then p_top
                         else if p_right.2 <= ymax && p_right.2 >= ymin then p_right
                         else pixelcenter

          let point2 = if horizontal && ray >= ymin && ray <= ymax then (xmax, ray)
                         else if vertical && ray >= xmin && ray <= xmax then (ray,ymax)
                         else if vertical || horizontal then pixelcenter
                         else if p_right.2 <= ymax && p_right.2 >= ymin then p_right
                         else if p_top.1 <= xmax && p_top.1 >= xmin then p_top
                         else if p_bottom.1 <= xmax && p_bottom.1 >= xmin then p_bottom
                         else if p_left.2 <= ymax && p_left.2 >= ymin then p_left
                         else pixelcenter

          in if point1.1 == pixelcenter.1 || point2.1 == pixelcenter.1 then 0 else distance point1 point2

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
