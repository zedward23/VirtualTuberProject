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

        public MainWindow()
        {
            InitializeComponent();
            eltsAdded = 1;
        }

        private void SpawnVertex(object sender, RoutedEventArgs e)
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

            Canvas.SetLeft(newRectangle, MyCanvas.ActualWidth / (2f));
            Canvas.SetTop(newRectangle, MyCanvas.ActualHeight / (2f));

            eltsAdded++;
            newRectangle.Tag = "added";
            newRectangle.Uid = ""+eltsAdded;

            MyCanvas.Children.Add(newRectangle);

            //
        }

        private void mouseMoving(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (e.OriginalSource is Rectangle)
            {
                Rectangle clicked = (Rectangle)e.OriginalSource;
                String tag = (String)clicked.Tag;
                selectedIdx = Int32.Parse(clicked.Uid);
                if (e.LeftButton == MouseButtonState.Pressed && tag.Equals("added"))
                {
                    DragDrop.DoDragDrop((Rectangle)MyCanvas.Children[selectedIdx], (Rectangle)MyCanvas.Children[selectedIdx], DragDropEffects.Move);
                }
            }
        }

        private void Dragging(object sender, System.Windows.DragEventArgs e)
        {
            Point dropPos = e.GetPosition(MyCanvas);

            Canvas.SetLeft((Rectangle)MyCanvas.Children[selectedIdx], dropPos.X);
            Canvas.SetTop((Rectangle)MyCanvas.Children[selectedIdx], dropPos.Y);
        }

        void drawTriangle(Vec2 pos1, Vec2 pos2, Vec2 pos3)
        {
            MyCanvas.Children.Add(drawLine(pos1, pos2));
            MyCanvas.Children.Add(drawLine(pos2, pos3));
            MyCanvas.Children.Add(drawLine(pos3, pos1));
        }

        Line drawLine(Vec2 pos1, Vec2 pos2)
        {
            Line l1 = new Line();

            l1.Stroke = System.Windows.Media.Brushes.Black;
            l1.Fill = System.Windows.Media.Brushes.Black;

            l1.X1 = pos1.x;
            l1.Y1 = pos1.y;

            l1.X2 = pos2.x;
            l1.Y2 = pos2.y;

            l1.Tag = "Line";

            return l1;
        }

        private void GenMesh(object sender, RoutedEventArgs e)
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

            var child1 = MyCanvas.Children[2];
            var child2 = MyCanvas.Children[3];
            var child3 = MyCanvas.Children[4];

            Point p1 = new Point(Canvas.GetLeft(child1), Canvas.GetTop(child1));
            Point p2 = new Point(Canvas.GetLeft(child2), Canvas.GetTop(child2));
            Point p3 = new Point(Canvas.GetLeft(child3), Canvas.GetTop(child3));

            Vec2 pos1 = new Vec2((float)p1.X, (float)p1.Y);
            Vec2 pos2 = new Vec2((float)p2.X, (float)p2.Y);
            Vec2 pos3 = new Vec2((float)p3.X, (float)p3.Y);

            drawTriangle(pos1, pos2, pos3);
        }
    }
}
