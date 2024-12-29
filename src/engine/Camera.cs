using System.Numerics;
namespace Engine;

public class Camera
{
    private Vector3 position;
    public Vector3 Position 
    {
        get => position;
        set
        {
            position = value;
            RecalculateMatrices();
        }
    }

    private Quaternion rotation;
    public Quaternion Rotation
    {
        get => rotation;
        set
        {
            rotation = value;
            RecalculateMatrices();
        }
    }

    public Vector3 Forward => Vector3.Transform(-Vector3.UnitZ, Rotation);
    public Vector3 Right => Vector3.Transform(Vector3.UnitX, Rotation);
    public Vector3 Up => Vector3.Transform(Vector3.UnitY, Rotation);


    public float FieldOfView {get; set;} = 60f;
    public float AspectRatio {get; set;} = 16f / 9f;
    public float Near {get; set;} = 0.01f;
    public float Far {get; set;} = 500f;

    public float FovY => FieldOfView / AspectRatio;
    public float FovX => FieldOfView;

    public Matrix4x4 ViewMatrix = Matrix4x4.Identity;
    public Matrix4x4 PerspectiveMatrix = Matrix4x4.Identity;
    public Matrix4x4 ViewProjectionMatrix = Matrix4x4.Identity;
    public Frustum Frustum {get; private set;} = new Frustum();

    public void RecalculateMatrices()
    {
        ViewMatrix = Matrix4x4.CreateLookAt(Position, Position + Forward, Up);
        PerspectiveMatrix = Matrix4x4.CreatePerspectiveFieldOfView(FieldOfView / 180f * MathF.PI, AspectRatio, Near, Far);
        ViewProjectionMatrix = ViewMatrix * PerspectiveMatrix;
        Frustum.CalculateFrustumPlanes(ViewProjectionMatrix);
    }


    public Camera(Vector3 position)
    {
        Position = position;
    }
}

