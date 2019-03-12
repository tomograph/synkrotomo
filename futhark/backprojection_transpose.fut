-- ==
-- input@../data/bpinputf32rad64
-- output@../data/bpoutputf32rad
-- input@../data/bpinputf32rad128
-- output@../data/bpoutputf32rad128
-- input@../data/bpinputf32rad256
-- input@../data/bpinputf32rad512
-- input@../data/bpinputf32rad1024
-- input@../data/bpinputf32rad1500
-- input@../data/bpinputf32rad2000
-- input@../data/bpinputf32rad2048
-- input@../data/bpinputf32rad3000
-- input@../data/bpinputf32rad3500
-- input@../data/bpinputf32rad4000
-- input@../data/bpinputf32rad4096


import "sirtlib"
open sirtlib

module bpTlib = {
  let fact = f32.sqrt(2.0f32)/2.0f32

  -- only works when lines have slope > 1. To use for all lines use preprocess to transpose lines and image
  let bp [p] [l] (lines: [l](f32, f32, f32))
            (rhozero: f32)
            (deltarho: f32)
            (rhosprpixel: i32)
            (numrhos: i32)
            (halfsize: i32)
            (projections: [p]f32) : []f32 =
            flatten <| map(\irow ->
                      map(\icolumn ->
                           let xmin = r32(icolumn)
                           let ymin = r32(irow)
                           in reduce (+) 0.0f32 <| map(\ln ->
                                let (cost, sint, lbase) = unsafe lines[ln]
                                let tant = sint/cost
                                let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                                let rho = cost*p.1+sint*p.2
                                let s = f32.ceil((rho-rhozero)/deltarho)
                                let xbase = ymin*tant
                                in reduce (+) 0.0f32 <| map(\i ->
                                     let sprime = s+(r32(i))
                                     let r = sprime*deltarho+rhozero
                                     let x_bot = (r/cost)-xbase
                                     let x_top = x_bot-tant
                                     let maxx = f32.max x_bot x_top
                                     let minx = f32.min x_bot x_top
                                     let l = (intersect_fact maxx minx xmin (xmin+1.0))*(lbase)
                                     let projectionidx = ln*numrhos+(t32(sprime))
                                     in l*(unsafe projections[projectionidx])
                                )(iota rhosprpixel)
                           ) (iota l)
                      )((-halfsize)...(halfsize-1))
                 )((-halfsize)...(halfsize-1))

  let backprojection (steep_projections: []f32) (flat_projections: []f32) (steep_lines: ([](f32, f32, f32))) (flat_lines: ([](f32, f32, f32))) (rhozero: f32) (deltarho: f32) (rhosprpixel: i32) (numrhos: i32) (halfsize: i32): []f32 =
        let bp_steep = bp steep_lines rhozero deltarho rhosprpixel numrhos halfsize steep_projections
        let bp_flat = bp flat_lines rhozero deltarho rhosprpixel numrhos halfsize flat_projections
        --untranspose in flat case
        let size = halfsize*2
        let bp_flatT =  if (size < 10000)
                     then flatten <| transpose <| unflatten size size bp_flat
                     else (replicate (size**2) 1.0f32)
        in map2 (+) bp_steep bp_flatT
}
open bpTlib

let main  [p][a](angles : [a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
          let halfsize = size/2
          let numrhos = p/a
          let (steep_lines, flat_lines, is_flat, _) = preprocess angles numrhos

	        let (steep_proj, flat_proj) = fix_projections projections is_flat
          in backprojection steep_proj flat_proj steep_lines flat_lines rhozero deltarho rhosprpixel numrhos halfsize
