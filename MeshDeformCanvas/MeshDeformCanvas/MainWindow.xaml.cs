using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Forms;

using DragDropEffects = System.Windows.DragDropEffects;

namespace MeshDeformCanvas
{

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>

    public partial class MainWindow : Window
    {
        int eltsAdded;
        int selectedIdx;

        Mesh mesh;

        public MainWindow()
        {
            InitializeComponent();
            eltsAdded = 1;
            mesh = new Mesh();

            Vec2 startPos = new Vec2(200f, 200);

            addVertex(startPos);
            addVertex(startPos.plus(new Vec2(50f, 0)));
            addVertex(startPos.plus(new Vec2(0f, 50f)));
            addVertex(startPos.plus(new Vec2(50f, 50f)));
            realTimeGen();
        }

        /*private void SpawnVertex(object sender, RoutedEventArgs e)
        {
            addVertex(new Vec2((float)MyCanvas.ActualWidth / (2f), (float)MyCanvas.ActualHeight / (2f)));
        }*/

        int canvasToMeshIdx(int canvasIdx)
        {
            Rectangle clicked = (Rectangle)MyCanvas.Children[canvasIdx];
            return Int32.Parse((String)clicked.Tag);
        }

        int meshToCanvasIdx(int meshIdx)
        {
            return mesh.vertices[meshIdx].canvasIdx;
        }

        private void mouseMoving(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (e.OriginalSource is Rectangle)
            {
                Rectangle clicked = (Rectangle)e.OriginalSource;
                selectedIdx = Int32.Parse(clicked.Uid);
                if (e.LeftButton == MouseButtonState.Pressed)
                {

                    int ad = selectedIdx;
                    var adfdafa = MyCanvas.Children;
                    DragDrop.DoDragDrop((Rectangle)MyCanvas.Children[selectedIdx], (Rectangle)MyCanvas.Children[selectedIdx], DragDropEffects.Move);
                }
            }
        }

        private void Dragging(object sender, System.Windows.DragEventArgs e)
        {
            Point dropPos = e.GetPosition(MyCanvas);

            Canvas.SetLeft((Rectangle)MyCanvas.Children[selectedIdx], dropPos.X);
            Canvas.SetTop((Rectangle)MyCanvas.Children[selectedIdx], dropPos.Y);

            int currMeshIdx = canvasToMeshIdx(selectedIdx);
            mesh.vertices[currMeshIdx].position = new Vec2((float)dropPos.X, (float)dropPos.Y);

            realTimeGen();


        }

        void addVertex(Vec2 pos)
        {
            mesh.vertices.Add(new Vertex(pos,
                                         MyCanvas.Children.Count,
                                         mesh.vertices.Count));
            drawVertex(mesh.vertices[mesh.vertices.Count - 1]);
            realTimeGen();
        }

        void drawVertex(Vertex p)
        {
            Brush c = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            Rectangle newRectangle = new Rectangle
            {
                Width = 5,
                Height = 5,
                Fill = c,
                StrokeThickness = 1,
                Stroke = Brushes.Black
            };

            Canvas.SetLeft(newRectangle, p.position.x);
            Canvas.SetTop(newRectangle, p.position.y);

            int meshIdx = mesh.vertices.Count-1;

            eltsAdded = MyCanvas.Children.Count;
            newRectangle.Tag = "" + meshIdx;
            newRectangle.Uid = "" + eltsAdded;

            MyCanvas.Children.Add(newRectangle);
        }

        void addLine(Vertex p1, Vertex p2)
        {
            mesh.edges.Add(new Edge(p1, p2, mesh.edges.Count, MyCanvas.Children.Count));
            drawLine(mesh.edges[mesh.edges.Count-1]);
        }

        void drawLine(Edge e)
        {

            Line l1 = new Line();

            l1.Stroke = System.Windows.Media.Brushes.Black;
            l1.Fill = System.Windows.Media.Brushes.Black;

            Vec2 pos1 = e.p1.position;
            Vec2 pos2 = e.p2.position;

            l1.X1 = pos1.x;
            l1.Y1 = pos1.y;

            l1.X2 = pos2.x;
            l1.Y2 = pos2.y;

            l1.Tag = "" + (mesh.edges.Count-1);
            l1.Uid = "" + MyCanvas.Children.Count;

            MyCanvas.Children.Add(l1);
        }

        /*private void GenMesh(object sender, RoutedEventArgs e)
        {
            List<UIElement> toBeRemoved = new List<UIElement>();

            foreach (UIElement elt in MyCanvas.Children)
            {
                if (elt is Line)
                {
                    toBeRemoved.Add(elt);
                }
            }

            foreach (UIElement elt in toBeRemoved)
            {
                if (elt is Line)
                {
                    MyCanvas.Children.Remove(elt);
                }
            }

            mesh.edges = new List<Edge>();

            for (int i = 0; i < MyCanvas.Children.Count; i++)
            {
                if (MyCanvas.Children[i] is Line)
                {
                    ((Line)MyCanvas.Children[i]).Uid = "" + i;
                }
                if (MyCanvas.Children[i] is Rectangle)
                {
                    ((Rectangle)MyCanvas.Children[i]).Uid = "" + i;
                }
            }

            for (int i = 0; i < mesh.vertices.Count; i++)
            {
                if (i >= 2)
                {
                    addLine(mesh.vertices[i], mesh.vertices[i - 1]);
                    addLine(mesh.vertices[i], mesh.vertices[i - 2]);
                } else if (i == 1)
                {
                    addLine(mesh.vertices[i], mesh.vertices[i - 1]);
                }
            }

        }*/

        void realTimeGen()
        {
            
            List<UIElement> toBeRemoved = new List<UIElement>();

            foreach (UIElement elt in MyCanvas.Children)
            {
                if (elt is Line)
                {
                    toBeRemoved.Add(elt);
                }
            }

            foreach (UIElement elt in toBeRemoved)
            {
                if (elt is Line)
                {
                    MyCanvas.Children.Remove(elt);
                }
            }

            mesh.edges = new List<Edge>();

            for (int i = 0; i < MyCanvas.Children.Count; i++)
            {
                if (MyCanvas.Children[i] is Line)
                {
                    ((Line)MyCanvas.Children[i]).Uid = "" + i;
                }
                if (MyCanvas.Children[i] is Rectangle)
                {
                    ((Rectangle)MyCanvas.Children[i]).Uid = "" + i;
                }
            }

            for (int i = 0; i < mesh.vertices.Count; i++)
            {
                if (i >= 2)
                {
                    addLine(mesh.vertices[i], mesh.vertices[i - 1]);
                    addLine(mesh.vertices[i], mesh.vertices[i - 2]);
                } else if (i == 1)
                {
                    addLine(mesh.vertices[i], mesh.vertices[i - 1]);
                }
            }
        }

        private void Clicked(object sender, MouseButtonEventArgs e)
        {
            Point dropPos = e.GetPosition(MyCanvas);
            if (Keyboard.IsKeyDown(Key.LeftShift))
            {
                addVertex(new Vec2((float)dropPos.X, (float)dropPos.Y));
            }
        }
    }
}
