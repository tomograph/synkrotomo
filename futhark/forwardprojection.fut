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
      let pixindexcorrection = halfsize*(1+size)
      in flatten <| map (\(cost, sint, lbase) ->
        -- determine slope in x = slope*y+intercept (in eq. rho = cost*x+sint*y divide by cost to get results)
        let slope = -sint/cost
        in map (\r ->
          let rho = rhozero + r32(r)*deltarho
          let intercept = rho/cost

          in reduce (+) 0.0f32 <| map(\i ->
            let xbot = r32(i)*slope + intercept
            let xtop = xbot+slope

            let Xpixbot = t32(f32.floor(xbot))
            let Xpixtop = t32(f32.floor(xtop))

            let Xpixmax = r32(i32.max Xpixbot Xpixtop)

            let xdiff = xtop - xbot

            let singlepixelcase = Xpixbot == Xpixtop
            let xbotwithinbounds = Xpixbot >= -halfsize && Xpixbot < halfsize
            let xtopwithinbounds = (!singlepixelcase) && Xpixtop >= -halfsize && Xpixtop < halfsize

            let lxbotfac = ((Xpixmax - xbot)/xdiff)
            let lxbot = if singlepixelcase then lbase else lxbotfac*lbase
            let lxtop = ((xtop - Xpixmax)/xdiff)*lbase

            let pixbot = Xpixbot+i*size+pixindexcorrection
            let pixtop = Xpixtop+i*size+pixindexcorrection

            let bot = if xbotwithinbounds then (unsafe lxbot*img[pixbot]) else 0.0f32
            let top = if xtopwithinbounds then (unsafe lxtop*img[pixtop]) else 0.0f32

            in (bot+top)
          ) ((-halfsize)...(halfsize-1))
        ) (iota numrhos)
      ) lines

  let forwardprojection (steep_lines: ([](f32, f32, f32))) (flat_lines: ([](f32, f32, f32))) (projection_indexes: []i32) (rhozero: f32) (deltarho: f32) (numrhos: i32) (halfsize: i32) (image: []f32) (imageT: []f32) : []f32 =
        let fp_steep = fp steep_lines rhozero deltarho numrhos halfsize image
        let fp_flat = fp flat_lines rhozero deltarho numrhos halfsize imageT
        -- if angles are in order steep - flat we might as well use fp_steep ++ fp_flat
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
  let imageT =  if (size < 10000)
                then flatten <| transpose <| copy (unflatten size size image)
                else (replicate n 1.0f32)

  in forwardprojection steep_lines flat_lines projection_indexes rhozero deltarho numrhos halfsize image imageT
