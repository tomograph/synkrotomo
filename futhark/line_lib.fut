module Lines = {
     type point  = ( f32, f32 )

     -- find x given y
     let find_x (y : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if cost == 0 then ray else (ray-y*sint)/cost

     -- find y given x
     let find_y (x : f32) (ray: f32) (cost: f32) (sint: f32): f32 =
          if sint == 0 then ray else (ray-x*cost)/sint

     -- gets entry and exit point in no particular order. might later consider corners and vertical lines on grid edge
     let entryexitPoint (sint : f32) (cost : f32) (ray : f32) (maxval : f32) : (point,point) =
          let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint) -- check if y is in grid
          let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval)) -- check if x is in grid
          let p_top = (find_x maxval ray cost sint, maxval) -- check if x is in grid
          let p_right = (maxval, find_y maxval ray cost sint) -- check if y is in grid

          let point1 = if sint==0 then (ray,-maxval) else if sint == 1 then (-maxval,ray)
               else if (f32.abs(p_left.2) <= maxval) then p_left
               else if (f32.abs(p_bottom.1) <= maxval) then p_bottom
               else if (f32.abs(p_top.1) <= maxval) then p_top
               else p_right

          let point2 = if sint==0 then (ray,maxval) else if sint == 1 then (maxval,ray)
               else if (f32.abs(p_right.2) <= maxval) then p_right
               else if (f32.abs(p_top.1) <= maxval) then p_top
               else if (f32.abs(p_bottom.1) <= maxval) then p_bottom
               else p_left

          in (point1, point2)

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
          let p_left = ((-1.0*maxval), find_y (-1.0*maxval) ray cost sint)
          let p_bottom = (find_x (-1.0*maxval) ray cost sint, (-1.0*maxval))
          let p_top = (find_x maxval ray cost sint, maxval)
          in if f32.abs(p_left.2) <= maxval && sint != 0 then p_left else if p_bottom.1 <= p_top.1 then p_bottom else p_top

     --convert to entry points !!!SHOULD ALSO return cost sint
     let convert2entry (angles: []f32) (rays: []f32) (maxval: f32): [] (point, point) =
          let sincos = convert2sincos angles
          let anglesrays = flatten(map (\t -> map(\r -> (t.1,t.2,r)) rays) sincos)
          in map(\(s,c,r) -> ((entryPoint s c r maxval), (s,c))) anglesrays

     -- let distance ((x1, y1) : point) ((x2, y2) : point): f32 =
     --      f32.sqrt( (x2 - x1) ** 2.0f32 + (y2 - y1) ** 2.0f32 )

     -- Check whether a point is within the grid.
     let isInGrid (halfsize : f32) (y_step_dir : f32) ((x, y) : point) : bool =
          x >= -1f32*halfsize && x < halfsize && (
               if y_step_dir == -1f32
               then (-1f32*halfsize < y && y <= halfsize)
               else (-1f32*halfsize <= y && y < halfsize)
          )

 }
