using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace MeshDeformCanvas
{
    class Vertex
    {
        public Vec2 position;
        public int canvasIdx;
        public int meshIdx;
        //System.Windows.UIElement drawnElt;

        public Vertex(Vec2 pos, int eltsAdded, int idx)
        {
            this.position = pos;
            this.canvasIdx = eltsAdded;
            this.meshIdx = idx;
        }
    }
}
