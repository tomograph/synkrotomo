-- ==
-- input@data/fpinputf32rad64
-- input@data/fpinputf32rad128
-- input@data/fpinputf32rad256
-- input@data/fpinputf32rad512
-- input@data/fpinputf32rad1024
-- input@data/fpinputf32rad1500
-- input@data/fpinputf32rad2000
-- input@data/fpinputf32rad2048
-- input@data/fpinputf32rad2500
-- input@data/fpinputf32rad3000
-- input@data/fpinputf32rad3500
-- input@data/fpinputf32rad4000
-- input@data/fpinputf32rad4096


import "sirtlib"
open sirtlib

module fpTlib = {
     let fp [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32) =
      let size = halfsize*2
      in flatten <| map (\(cos, sin, lbase) ->
        let k = sin/cos
        in map (\r ->
          let rho = rhozero + r32(r)*deltarho
          let base = rho/cos

          in reduce (+) 0.0f32 <| map(\i ->
            let ih = i+halfsize

            let xmin = base-r32(i)*k + (r32(halfsize))
            let xplus = xmin-k + (r32(halfsize))

            let Xpixmin = t32(f32.floor(xmin))
            let Xpixplus = t32(f32.floor(xplus))

            let Xpixmax = r32(i32.max Xpixmin Xpixplus)

            let xdiff = xplus - xmin

            let bounds = ih >= 0 && ih < size
            let eq = Xpixmin == Xpixplus
            let bmin = bounds && Xpixmin >= 0 && Xpixmin < size
            let bplus = (!eq) && bounds && Xpixplus >= 0 && Xpixplus < size

            let lxminfac = ((Xpixmax - xmin)/xdiff)
            let lxmin = if eq then lbase else lxminfac*lbase
            let lxplus = ((xplus - Xpixmax)/xdiff)*lbase

            let pixmin = Xpixmin+ih*size
            let pixplus = Xpixplus+ih*size

            let min = if bmin then (unsafe lxmin*img[pixmin]) else 0.0f32
            let plus = if bplus then (unsafe lxplus*img[pixplus]) else 0.0f32

            in (min+plus)
          ) ((-halfsize)...(halfsize-1))
        ) (iota numrhos)
      ) lines

  let forwardprojection (steep_lines: ([](f32, f32, f32))) (flat_lines: ([](f32, f32, f32))) (projection_indexes: []i32) (rhozero: f32) (deltarho: f32) (numrhos: i32) (halfsize: i32) (image: []f32) (imageT: []f32) : []f32 =
        let fp_steep = fp steep_lines rhozero deltarho numrhos halfsize image
        let fp_flat = fp flat_lines rhozero deltarho numrhos halfsize imageT
        in postprocess_fp projection_indexes fp_steep fp_flat

}

open fpTlib

let main  [n][a] (angles : *[a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (numrhos : i32)
          (image : *[n]f32) =
  let size = t32(f32.sqrt(r32(n)))
  let halfsize = size/2

  let (steep_lines, flat_lines, _, projection_indexes) = preprocess angles numrhos
  -- hack to always do this!
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  in forwardprojection steep_lines flat_lines projection_indexes rhozero deltarho numrhos halfsize image imageT
