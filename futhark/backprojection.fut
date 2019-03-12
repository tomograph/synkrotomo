-- ==
-- input@data/bpinputf32rad64
-- input@data/bpinputf32rad128
-- input@data/bpinputf32rad256
-- input@data/bpinputf32rad512
-- input@data/bpinputf32rad1024
-- input@data/bpinputf32rad1500
-- input@data/bpinputf32rad2000
-- input@data/bpinputf32rad2048
-- input@data/bpinputf32rad3000
-- input@data/bpinputf32rad3500
-- input@data/bpinputf32rad4000
-- input@data/bpinputf32rad4096

import "sirtlib"
open sirtlib

module bplib = {

  let preprocess_2 [a](angles: [a]f32): ([](f32,f32,f32,i32),[](f32,f32,f32,i32)) =
       let cossin = map(\i -> let angle = angles[i]
            let cos= f32.cos(angle)
            let sin = f32.sin(angle)
            let lcot  = f32.sqrt(1.0+(cos/sin)**2.0f32)
            let ltan = f32.sqrt(1.0+(sin/cos)**2.0f32)
            in (cos, sin, lcot,ltan, i))
       (iota(a))
       let parts = partition(\(c,s,_,_,_) -> is_flat c s  )cossin
       in ((map(\(cos, sin, lcot,_, i)-> (cos,-sin,lcot,i))parts.1),(map(\(cos, sin, _,ltan, i)-> (cos,-sin,ltan,i))parts.2))

  let bp_steep [p] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (rhosprpixel: i32) (numrhos: i32) (halfsize: i32) (projections: [p]f32): []f32 =
       let fact = f32.sqrt(2.0f32)/2.0f32
       in flatten (map(\irow ->
               map(\icolumn ->
                     let xmin = r32(icolumn)
                     let ymin = r32(irow)
                     in reduce (+) 0.0f32 <| map(\(cost,sint,lbase,angleidx) ->
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
                                    let projectionidx = angleidx*numrhos+(t32(sprime))
                                    in l*(unsafe projections[projectionidx])
                               )(iota rhosprpixel)
                     )lines
           )((-halfsize)...(halfsize-1))
      )((-halfsize)...(halfsize-1)))

   let bp_flat [p] (lines: ([](f32,f32,f32,i32))) (rhozero: f32) (deltarho: f32) (rhosprpixel: i32) (numrhos: i32) (halfsize: i32) (projections: [p]f32): []f32 =
         let fact = f32.sqrt(2.0f32)/2.0f32
         in flatten (map(\irow ->
            map(\icolumn ->
                 let xmin = r32(icolumn)
                 let ymin = r32(irow)
                 in reduce (+) 0.0f32 <| map(\(cost,sint,lbase,angleidx) ->
                      let cott = cost/sint
                      let p = (xmin+0.5f32-fact*cost, ymin+0.5f32-fact*sint)
                      let rho = cost*p.1+sint*p.2
                      let s = f32.ceil((rho-rhozero)/deltarho)
                      let ybase = xmin*cott
                      in reduce (+) 0.0f32 <| map(\i ->
                                let sprime = s+(r32(i))
                                let r = sprime*deltarho+rhozero
                                let y_left = (r/sint)-ybase
                                let y_right = y_left-cott
                                let maxy = f32.max y_left y_right
                                let miny = f32.min y_left y_right
                                let l = (intersect_fact maxy miny ymin (ymin+1.0))*lbase
                                let projectionidx = angleidx*numrhos+(t32(sprime))
                                in l*(unsafe projections[projectionidx])
                           )(iota rhosprpixel)
                      )lines
                 )((-halfsize)...(halfsize-1))
            )((-halfsize)...(halfsize-1)))

  let backprojection [p][a](angles : [a]f32)
            (rhozero : f32)
            (deltarho : f32)
            (size : i32)
            (projections: [p]f32): []f32 =
            let rhosprpixel = t32(f32.ceil(f32.sqrt(2)/deltarho))
            let halfsize = size/2
            let lines = preprocess_2 angles
            let numrhos = p/a
            let steep = bp_steep lines.2 rhozero deltarho rhosprpixel numrhos halfsize projections
            let flat = bp_flat lines.1 rhozero deltarho rhosprpixel numrhos halfsize projections
            in map2 (+) steep flat
}

open bplib

let main [p][a](angles : [a]f32)
          (rhozero : f32)
          (deltarho : f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          backprojection angles rhozero deltarho size projections
