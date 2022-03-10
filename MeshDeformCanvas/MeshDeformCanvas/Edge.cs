using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MeshDeformCanvas
{
    class Edge
    {
        public Vertex p1;
        public Vertex p2;
        public int meshIdx;
        public int canvasIdx;

        public Edge(Vertex q1, Vertex q2, int mesh, int canvas) 
        {
            this.p1 = q1;
            this.p2 = q2;
            meshIdx = mesh;
            canvasIdx = canvas;
        }
    }
}
