

using System.Numerics;
using Silk.NET.Input;
namespace Engine;

public class FreeLookCamera : Camera
{
    public float mouseSensitivity = 0.3f;
    public float speed = 50f;
    private Vector2 lastMousePosition;

    public Vector3 yawPitchRoll;
    private IKeyboard Keyboard;
    private IMouse Mouse;

    public FreeLookCamera(IKeyboard keyboard, IMouse mouse, Vector3 position, Vector3? yawPitchRoll = null) : base(position)
    {
        if (yawPitchRoll != null)
            this.yawPitchRoll = yawPitchRoll.Value;

        Keyboard = keyboard;
        Mouse = mouse;
    }

    public void Update(double deltaTime)
    {
        var forward = Forward;
        var right = Right;
        var up = Up;

        if (Keyboard.IsKeyPressed(Key.W))
            Position += forward * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.S))
            Position -= forward * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.D))
            Position += right * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.A))
            Position -= right * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.Space))
            Position += up * (float)deltaTime * speed;
        if (Keyboard.IsKeyPressed(Key.ShiftLeft))
            Position -= up * (float)deltaTime * speed;

        if (lastMousePosition == Vector2.Zero)
            lastMousePosition = Mouse.Position;

        var mouseDelta = Mouse.Position - lastMousePosition;
        yawPitchRoll.X -= mouseDelta.X * mouseSensitivity * (float)deltaTime;
        yawPitchRoll.Y -= mouseDelta.Y * mouseSensitivity * (float)deltaTime;

        Rotation = Quaternion.CreateFromYawPitchRoll(yawPitchRoll.X, yawPitchRoll.Y, yawPitchRoll.Z);

        lastMousePosition = Mouse.Position;
    }
}