using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
namespace Engine;

public class PointCloudVisualizer : GameWindow
{
    private List<Vector3> points;
    private float rotationSpeed = 0.5f;  // Speed of camera orbit
    private Vector3 cloudCenter;
    private float cloudRadius;
    private int vertexBufferObject;
    private int vertexArrayObject;
    private int shaderProgram;
    private Matrix4 projection;
    private float angle = 0.0f;
    private float pointRadius;

    public PointCloudVisualizer(List<System.Numerics.Vector3> points, float pointRadius)
        : base(GameWindowSettings.Default, new NativeWindowSettings { Size = new Vector2i(800, 600), Title = "3D Point Cloud Visualizer" })
    {
        this.points = points.ConvertAll(p => new Vector3(p.X, p.Y, p.Z));
        this.pointRadius = pointRadius;
        FitViewToPoints();
    }

    protected override void OnLoad()
    {
        base.OnLoad();

        GL.ClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        GL.Enable(EnableCap.DepthTest);

        // Create and bind VAO and VBO
        vertexArrayObject = GL.GenVertexArray();
        GL.BindVertexArray(vertexArrayObject);

        vertexBufferObject = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, vertexBufferObject);
        GL.BufferData(BufferTarget.ArrayBuffer, points.Count * Vector3.SizeInBytes, points.ToArray(), BufferUsageHint.StaticDraw);

        // Load shaders
        string vertexShaderSource = @"
            #version 330 core
            layout (location = 0) in vec3 aPosition;
            uniform mat4 uModelViewProjection;
            void main()
            {
                gl_Position = uModelViewProjection * vec4(aPosition, 1.0);
                gl_PointSize = 10.0; // Set point size dynamically later
            }
        ";

        string fragmentShaderSource = @"
            #version 330 core
            out vec4 FragColor;
            void main()
            {
                // Calculate the distance from the center of the point
                float dist = length(gl_PointCoord - vec2(0.5, 0.5));

                // If the distance is greater than 0.5, discard the fragment (make the point circular)
                if (dist > 0.5)
                    discard;

                FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // Set point color to white
            }
        ";

        // Compile and link shaders
        int vertexShader = CompileShader(ShaderType.VertexShader, vertexShaderSource);
        int fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentShaderSource);
        shaderProgram = GL.CreateProgram();
        GL.AttachShader(shaderProgram, vertexShader);
        GL.AttachShader(shaderProgram, fragmentShader);
        GL.LinkProgram(shaderProgram);

        // Clean up shaders (they are linked into the program and not needed anymore)
        GL.DetachShader(shaderProgram, vertexShader);
        GL.DetachShader(shaderProgram, fragmentShader);
        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        // Specify the layout of the vertex data
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, Vector3.SizeInBytes, 0);
        GL.EnableVertexAttribArray(0);
    }

    protected override void OnResize(ResizeEventArgs e)
    {
        base.OnResize(e);
        GL.Viewport(0, 0, Size.X, Size.Y);
        projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.PiOver4, Size.X / (float)Size.Y, 0.1f, 1000.0f);
    }

    private void FitViewToPoints()
    {
        // Calculate bounding box to center and scale the view
        Vector3 min = new Vector3(float.MaxValue);
        Vector3 max = new Vector3(float.MinValue);

        foreach (var point in points)
        {
            min = Vector3.ComponentMin(min, point);
            max = Vector3.ComponentMax(max, point);
        }

        cloudCenter = (min + max) / 2.0f;
        cloudRadius = (max - min).Length / 2.0f;
    }

    protected override void OnRenderFrame(FrameEventArgs e)
    {
        base.OnRenderFrame(e);

        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        // Calculate view matrix (orbiting around the point cloud)
        angle += (float)e.Time * rotationSpeed;
        Vector3 cameraPos = new Vector3(MathF.Cos(angle), 0.5f, MathF.Sin(angle)) * cloudRadius * 2 + cloudCenter;
        Matrix4 view = Matrix4.LookAt(cameraPos, cloudCenter, Vector3.UnitY);

        // Calculate the model-view-projection matrix
        Matrix4 modelViewProjection = view * projection;

        // Render the point cloud
        GL.UseProgram(shaderProgram);

        int mvpLocation = GL.GetUniformLocation(shaderProgram, "uModelViewProjection");
        GL.UniformMatrix4(mvpLocation, false, ref modelViewProjection);

        // Set point size using the radius
        GL.PointSize(pointRadius);

        GL.BindVertexArray(vertexArrayObject);
        GL.DrawArrays(PrimitiveType.Points, 0, points.Count);

        SwapBuffers();
    }

    private int CompileShader(ShaderType type, string source)
    {
        int shader = GL.CreateShader(type);
        GL.ShaderSource(shader, source);
        GL.CompileShader(shader);

        // Check for compilation errors
        GL.GetShader(shader, ShaderParameter.CompileStatus, out int success);
        if (success == 0)
        {
            string infoLog = GL.GetShaderInfoLog(shader);
            throw new Exception($"Error compiling {type}: {infoLog}");
        }

        return shader;
    }

    protected override void OnUnload()
    {
        base.OnUnload();

        // Cleanup
        GL.DeleteBuffer(vertexBufferObject);
        GL.DeleteVertexArray(vertexArrayObject);
        GL.DeleteProgram(shaderProgram);
    }
}