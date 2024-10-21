using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;
using SilkWindowObj = Silk.NET.Windowing.Window;

namespace Engine;

public class Window
{
    public static Window MainWindow { get; private set; }

    private IWindow window;
    private List<Action<double>> renderActions = new();
    private List<Action<double>> updateActions = new();
    public GLContext context;
    protected GL gl => context.gl;
    public Vector2D<int> Size => window.Size;

    public IWindow SilkWindow => window;

    
    public void OnRender(Action<double> action)
    {
        renderActions.Add(action);
    }

    public void OnUpdate(Action<double> action)
    {
        updateActions.Add(action);
    }



    public Window()
    {
        if (MainWindow == null)
        {
            MainWindow = this;
        }

        WindowOptions options = WindowOptions.Default;
        options.Size = new Vector2D<int>(800, 600);
        options.Title = "1.2 - Drawing a Quad";

        window = SilkWindowObj.Create(options);
        context = new GLContext(window);

        window.Render += Render;
        window.Update += Update;
        window.Resize += (size) =>
        {
            gl.Viewport(0, 0, (uint)size.X, (uint)size.Y);
        };

        window.Load += () =>
        {
            // gl.Disable(GLEnum.Blend);
            // gl.BlendFunc(GLEnum.SrcAlpha, GLEnum.OneMinusSrcAlpha);
            // gl.Disable(GLEnum.DepthTest);
            gl.ClearColor(0.2f,0.3f,0.3f, 1.0f);
            // gl.Disable(GLEnum.CullFace);
            // gl.CullFace(GLEnum.Back);
            gl.Viewport(0, 0, (uint)options.Size.X, (uint)options.Size.Y);
        };
    }

    public async Task Load()
    {
        context.Begin();
        while (!context.IsLoaded)
        {
            await Task.Delay(1);
        }
    }

    protected void Render(double deltaTime)
    {
        while (context.renderQueue.TryDequeue(out var action))
        {
            action.Invoke(deltaTime, gl);
        }

        gl.Clear((uint)ClearBufferMask.ColorBufferBit | (uint)ClearBufferMask.DepthBufferBit);

        foreach (var action in renderActions)
        {
            action.Invoke(deltaTime);
        }
    }

    protected void Update(double deltaTime)
    {
        while (context.updateQueue.TryDequeue(out var action))
        {
            action.Invoke(deltaTime, gl);
        }

        foreach (var action in updateActions)
        {
            action.Invoke(deltaTime);
        }
    }


}