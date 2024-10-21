using System.Collections.Concurrent;
using Silk.NET.Input;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;

namespace Engine;

public class GLContext
{
    private GL _gl;
    private IInputContext _input;
    private Thread glThread;
    private double _currentDt;
    private double _time;
    private bool _loaded = false;

    internal ConcurrentQueue<Action<double, GL>> renderQueue = [];
    internal ConcurrentQueue<Action<double, GL>> updateQueue = [];

    public GL gl => _gl;
    public IInputContext Input => _input;
    public double CurentDeltaTime => _currentDt;
    public double Time => _time;
    public bool IsLoaded => _loaded;


    public GLContext(IWindow window)
    {
        window.Load += () =>
        {
            _gl = window.CreateOpenGL();
            _input = window.CreateInput();
            _loaded = true;
            Console.WriteLine("OpenGL Loaded: " + gl.GetStringS(GLEnum.Version));
        };

        window.Render += (dt) => 
        {
            _currentDt = dt;
            _time += dt;
        };

        glThread = new Thread(window.Run);
    }

    public void Begin()
    {
        glThread.Start();
    }

    public void ExecuteCmd(Action<double, GL> action)
    {
        if (Thread.CurrentThread == glThread)
        {
            action(CurentDeltaTime, gl);
        }
        updateQueue.Enqueue(action);
    }

    public void RenderCmd(Action<double, GL> action)
    {
        if (Thread.CurrentThread == glThread)
        {
            action(CurentDeltaTime, gl);

            
        }
        renderQueue.Enqueue(action);
    }

    public static implicit operator GL(GLContext c) => c.gl;
}