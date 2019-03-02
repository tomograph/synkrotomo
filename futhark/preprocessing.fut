let is_flat (cos: f32) (sin: f32): bool =
     f32.abs(sin) >= f32.abs(cos)

let preprocessing [a][r] (angles : [a]f32)
          (rhos : [r]f32) =
          let cossin = map(\i -> let angle = angles[i]
               let cos= f32.cos(angle)
               let sin = f32.sin(angle)
               let lcot  = f32.sqrt(1.0+(cos/sin)**2.0f32)
               let ltan = f32.sqrt(1.0+(sin/cos)**2.0f32)
               in (cos, sin, lcot,ltan, i))
          (iota(a))
          let parts = partition(\(c,s,_,_,_) -> is_flat c s )cossin
          let lines = ((map (\(cos, sin, lcot,_, i)-> (cos,sin,lcot,i)) parts.1), (map(\(cos, sin, _,ltan, i)-> (cos,sin,ltan,i)) parts.2))
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          in (lines, rhozero, deltarho, r)
