

using System.Numerics;
using Silk.NET.Input;
namespace Engine;

public class FreeLookCamera : Camera
{
    public float mouseSensitivity = 0.3f;
    public float speed = 50f;
    protected Vector2 lastMousePosition;

    public Vector3 yawPitchRoll;
    protected IKeyboard Keyboard;
    protected IMouse Mouse;

    public bool handleMove = true;
    public bool handleLook = true;

    public FreeLookCamera(IKeyboard keyboard, IMouse mouse, Vector3 position, Vector3? yawPitchRoll = null) : base(position)
    {
        if (yawPitchRoll != null)
            this.yawPitchRoll = yawPitchRoll.Value;

        Keyboard = keyboard;
        Mouse = mouse;
    }

    public void HandleMove(double deltaTime)
    {
        if (Keyboard.IsKeyPressed(Key.W))
                Position += Forward * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.S))
            Position -= Forward * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.D))
            Position += Right * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.A))
            Position -= Right * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.Space))
            Position += Up * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.ShiftLeft))
            Position -= Up * (float)deltaTime * speed;
    }

    public void HandleLook(double deltaTime)
    {
        if (lastMousePosition == Vector2.Zero)
            lastMousePosition = Mouse.Position;

        var mouseDelta = Mouse.Position - lastMousePosition;
        yawPitchRoll.X -= mouseDelta.X * mouseSensitivity * (float)deltaTime;
        yawPitchRoll.Y -= mouseDelta.Y * mouseSensitivity * (float)deltaTime;

        Rotation = Quaternion.CreateFromYawPitchRoll(yawPitchRoll.X, yawPitchRoll.Y, yawPitchRoll.Z);

        lastMousePosition = Mouse.Position;
    }

    public virtual void Update(double deltaTime)
    {
        if (handleMove) HandleMove(deltaTime);        
        if (handleLook) HandleLook(deltaTime);
    }
}