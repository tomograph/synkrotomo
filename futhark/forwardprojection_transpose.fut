-- ==
-- input@../data/fpinputf32rad64
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad2048


import "sirtlib"
open sirtlib

module fpTlib = {
  let intersect_steep (x_bot: f32) (x_top: f32) (baselength: f32): ((f32,i32),(f32,i32)) =
       let Xpixmin = t32(f32.floor(x_bot))
       let Xpixplus = t32(f32.floor(x_top))
       let Xpixmax = i32.max Xpixmin Xpixplus
       let xdiff = x_top - x_bot
       -- if both equal then l is baselength and we only want one l
       let xminfact = if Xpixmin == Xpixplus then 1 else (r32(Xpixmax) - x_bot)/xdiff
       let xplusfact = if Xpixmin == Xpixplus then 0 else (x_top - r32(Xpixmax))/xdiff
       let lxmin = xminfact*baselength
       let lxplus = xplusfact*baselength
       in ((lxmin, Xpixmin), (lxplus, Xpixplus))

  let calculate_product [n]
           (x_bot: f32)
           (x_top: f32)
           (y: i32)
           (baselength: f32)
           (halfsize: i32)
           (vct: [n]f32) : f32 =
       let ((lmin,xmin),(lplus,xplus)) = intersect_steep x_bot x_top baselength
       let size = halfsize*2
       let ymin = y+halfsize
       let yplus = ymin+1
       let pixmin = xmin+ymin*size
       let pixplus = xplus+yplus*size
       let min = if xmin >= 0 && xmin < size && ymin >=0 && ymin < size then (unsafe lmin*vct[pixmin]) else 0.0f32
       let plus = if  xplus >= 0 && xplus < size && yplus >=0 && yplus < size then (unsafe lplus*vct[pixplus]) else 0.0f32
       in (min+plus)

  -- calculate one value in the forward projection vector
  let forward_projection_value (tant: f32) (baselength: f32) (rhoadjust: f32) (halfsize: i32) (img: []f32): f32 =
       reduce (+) 0.0f32 <| map(\ymin ->
               let x_bot = rhoadjust-r32(ymin)*tant
               let x_top = x_bot-tant
                in calculate_product x_bot x_top ymin baselength halfsize img
              )((-halfsize)...(halfsize-1))

  let fp [n] (lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (numrhos:i32) (halfsize: i32) (img: [n]f32): []f32 =
       flatten <| map(\(cos, sin, baselength)->
            let tant = sin/cos
            in map(\r ->
                 let rho = rhozero + r32(r)*deltarho
                 let rhoadjust = rho/cos
                 in forward_projection_value tant baselength rhoadjust halfsize img
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
