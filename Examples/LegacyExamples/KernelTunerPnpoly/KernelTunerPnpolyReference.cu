__global__ void PnpolyReference(int* bitmap, float2* points, float2* vertices, int n, int VERTICES) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int c = 0;
        float2 p = points[i];

        int k = VERTICES-1;

        for (int j=0; j<VERTICES; k = j++) {    // edge from v to vp
            float2 vj = vertices[j]; 
            float2 vk = vertices[k]; 

            float slope = (vk.x-vj.x) / (vk.y-vj.y);

            if ( (  (vj.y>p.y) != (vk.y>p.y)) &&            //if p is between vj and vk vertically
                    (p.x < slope * (p.y-vj.y) + vj.x) ) {   //if p.x crosses the line vj-vk when moved in positive x-direction
                c = !c;
            }
        }

        bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
    }
}

