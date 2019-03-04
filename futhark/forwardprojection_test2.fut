-- ==
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad2048

-- import "projection_lib"
-- open Projection
-- import "preprocessing"

let is_flat (cos: f32) (sin: f32): bool =
  f32.abs(sin) >= f32.abs(cos)

let preprocess_2 [a] (angles: [a]f32): ([](f32, f32, f32, i32), [](f32, f32, f32, i32)) =
  let cossin = map (\i ->
    let angle = angles[i]
    let cos= f32.cos(angle)
    let sin = f32.sin(angle)
    let lcot  = f32.sqrt(1.0 + (cos/sin)**2.0f32)
    let ltan = f32.sqrt(1.0 + (sin/cos)**2.0f32)
    in (cos, sin, lcot, ltan, i)
  ) (iota(a))
  let parts = partition (\(c, s, _, _, _) -> is_flat c s ) cossin
  let p1 = map (\(cos, sin, lcot, _, i) -> (cos, sin, lcot, i)) parts.1
  let p2 = map (\(cos, sin, _, ltan, i)-> (cos, sin, ltan, i)) parts.2
  in (p1, p2)

let forwardprojection_steep [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
  let fhalfsize = r32(halfsize)
  let size = halfsize*2
  in flatten <| map (\(cos, sin, lbase, _) ->
    let entBase = ((-fhalfsize)*sin)/cos
    let extBase = (fhalfsize*sin)/cos
    in map (\r ->
      let rho = (rhozero + r32(r)*deltarho)
      let ent = (rho - entBase, (-fhalfsize))
      let ext = (rho - extBase, fhalfsize)
      let k = (ext.1 - ent.1)/(ext.2 - ent.2)

      in (reduce (+) 0.0f32 fpv) <| map (\i ->
        let xmin = k*(r32(i) - ent.2) + ent.1 + (fhalfsize)
        let xplus = k*(r32(i) + 1 - ent.2) + ent.1 + (fhalfsize)
        let xdiff = xplus - xmin

        let Xpixmin = f32.floor(xmin)
        let Xpixplus = f32.floor(xplus)
        let Xpixmax = f32.max Xpixmin Xpixplus

        let bounds = (i+halfsize) >= 0 && (i+halfsize) < size
        let b = f32.abs(Xpixmin - Xpixplus) < 0.0005f32

        let bmin = bounds && Xpixmin >= 0.0f32 && Xpixmin < r32(size)
        let bplus = (!b) && bounds && Xpixplus >= 0.0f32 && Xpixplus < r32(size)

        let xminfacttmp = (Xpixmax - xmin)/xdiff
        let xminfact = if b then 1 else xminfacttmp
        let xplusfact = (xplus - Xpixmax)/xdiff

        let lxmin = xminfact*lbase
        let lxplus = xplusfact*lbase

        -- test manipulating the index and always reading a value, maybe on index 0 if out of bounds
        let min = if bmin then lxmin*(unsafe img[t32(Xpixmin)+(i+halfsize)*size]) else 0.0f32
        let plus = if bplus then lxplus*(unsafe img[t32(Xpixplus)+(i+halfsize)*size]) else 0.0f32

        in (min+plus)
      ) ((-halfsize)...(halfsize-1))
    ) (iota numrhos)
  ) lines

let forwardprojection_flat [n] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
  let fhalfsize = r32(halfsize)
  let size = halfsize*2
  in flatten <| map (\(cos, sin, lbase, _) ->
  let entBase = ((-fhalfsize)*cos)/sin
  let extBase = (fhalfsize*cos)/sin
  in map (\r ->
    let rho = rhozero + r32(r)*deltarho
    let ent = ((-fhalfsize), rho - entBase)
    let ext = (fhalfsize, rho - extBase)
    let k = (ext.2 - ent.2)/(ext.1 - ent.1)

    in (reduce (+) 0.0f32 fpv) <| map (\i ->
      let ymin = k*(r32(i) - ent.1) + ent.2 + fhalfsize
      let yplus = k*(r32(i) + 1 - ent.1) + ent.2 + fhalfsize
      let ydiff = yplus - ymin

      let Ypixmin = f32.floor(ymin)
      let Ypixplus = f32.floor(yplus)
      let Ypixmax = f32.max Ypixmin Ypixplus

      let bounds = (i+halfsize) >= 0 && (i+halfsize) < size

      let b = f32.abs(Ypixmin - Ypixplus) < 0.0005f32
      let bmin = bounds && Ypixmin >= 0.0f32 && Ypixmin < r32(size)
      let bplus = (!b) && bounds && Ypixplus >= 0.0f32 && Ypixplus < r32(size)

      let yminfacttmp = (Ypixmax - ymin)/ydiff
      let yminfact = if b then 1 else yminfacttmp
      let yplusfact = (yplus - Ypixmax)/ydiff

      let lymin = yminfact*lbase
      let lyplus = yplusfact*lbase

      let min = if bmin then lymin*(unsafe img[(i+halfsize)+t32(Ypixmin)*size]) else 0.0f32
      let plus = if bplus then lyplus*(unsafe img[(i+halfsize)+t32(Ypixplus)*size]) else 0.0f32

      in (min+plus)
    ) ((-halfsize)...(halfsize-1))
  ) (iota numrhos)
) lines

let main  [n][r] (angles : []f32)
          (rhos : [r]f32)
          (image : *[n]f32) =
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2
  let rhozero = unsafe rhos[0]
  let deltarho = unsafe rhos[1]-rhozero
  let numrhos = r
  let lines = preprocess_2 angles
  -- let (lines, rhozero, deltarho, numrhos) = preprocessing angles rhos
  let steep = forwardprojection_steep lines.2 rhozero deltarho numrhos halfsize image
  let flat = forwardprojection_flat lines.1 rhozero deltarho numrhos halfsize image
  in steep ++ flat
