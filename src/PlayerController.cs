using System.Numerics;
using Engine;
using Silk.NET.Input;

public class PlayerController
{
    public Vector3 physicsVelocity;
    public Vector3 movementVelocity;
    public Vector3 lastMoveDirection;
    public float jumpImpulse = 4f;
    public float acceleration = 5f;
    public float runAcceleration = 8f;
    public float deceleration = 10f;
    public float walkSpeed = 1.25f;
    public float runSpeed = 3.0f;
    public float heightLerpSpeed = 25f;

    // Crouch parameters
    public float crouchSpeedMultiplier = 0.3f;
    public float crouchHeightMultiplier = 0.3f;
    private bool isCrouching = false;
    private float currentHeightMultiplier = 1f;

    // Coyote time and jump buffer parameters
    private float coyoteTimeSeconds = 0.15f;
    private float jumpBufferSeconds = 0.15f;
    private float coyoteTimer = 0;
    private float jumpBufferTimer = 0;
    private bool hasJumped = false;
    private bool wasGrounded = false;

    // Step height parameters
    public float maxStepHeight = 0.08f; // Maximum height of step that can be climbed
    private bool isStepTooHigh = false;
    public float currentStepHeight = 0f;

    protected VoxelRenderer voxelWorld;
    public bool IsGrounded { get; private set; } = false;
    
    public RaycastHit groundHit;
    public RaycastHit frontHit;
    public float playerHeight = 0.25f;
    
    private Vector3 position;
    public Vector3 Position 
    { 
        get => position;
        private set => position = value;
    }
    
    public Vector3 VisualPosition { get; private set; }
    
    public FreeLookCamera Camera { get; private set; }
    protected IKeyboard Keyboard;
    protected IMouse Mouse;

    // freeze for a time so the world can load 
    public float freezeTime = 2.0f;

    public PlayerController(IKeyboard keyboard, IMouse mouse, Vector3 position, Vector3? cameraRotation = null)
    {
        this.Position = position;
        this.VisualPosition = position;
        
        Keyboard = keyboard;
        Mouse = mouse;
        
        Camera = new FreeLookCamera(keyboard, mouse, position, cameraRotation)
        {
            handleMove = false
        };

        Keyboard.KeyDown += (board, key, arg3) =>
        {
            if (key == Key.Space && !isCrouching)
            {
                if (IsGrounded || coyoteTimer > 0)
                {
                    physicsVelocity.Y = jumpImpulse;
                    hasJumped = true;
                    coyoteTimer = 0;
                    jumpBufferTimer = 0;
                }
                else
                {
                    jumpBufferTimer = jumpBufferSeconds;
                }
            }
        };

        Keyboard.KeyUp += (board, key, arg3) =>
        {
            if (key == Key.Space && physicsVelocity.Y > 0)
            {
                physicsVelocity.Y /= 2;
            }
        };
    }

    public void SetVoxelWorld(VoxelRenderer voxelWorld)
    {
        this.voxelWorld = voxelWorld;
    }

    public void Update(double deltaTime)
    {
        float dt = (float)deltaTime;
        if (voxelWorld == null) throw new System.Exception("PlayerController requires a VoxelRenderer to be set");

        if (freezeTime > 0)
        {
            freezeTime -= dt;
            return;
        }
        
        // Update timers
        if (coyoteTimer > 0) coyoteTimer = Math.Max(0, coyoteTimer - dt);
        if (jumpBufferTimer > 0) jumpBufferTimer = Math.Max(0, jumpBufferTimer - dt);
        
        // Update physics first
        physicsVelocity += Vector3.UnitY * -6.8f * dt;
        
        // Ground check at new position
        var randomVariation = new Vector3(Random.Shared.NextSingle() * 0.0001f);
        var groundCast = new RaycastInput(Position, -Vector3.UnitY + randomVariation, playerHeight - physicsVelocity.Y * dt);
        var frontCast = new RaycastInput(Position + lastMoveDirection * 0.15f, -Vector3.UnitY + randomVariation, playerHeight);
        RaycastResult[] results = voxelWorld.raycastSystem.RaycastBatch(new[] { groundCast, frontCast }).ToArray();
        groundHit = results[0].hits[0];
        frontHit = results[1].hits[0];
        IsGrounded = groundHit.valid;

        // Check if step is too high
        isStepTooHigh = false;
        if (IsGrounded && frontHit.valid)
        {
            currentStepHeight = frontHit.position.Y - groundHit.position.Y;
            isStepTooHigh = currentStepHeight > maxStepHeight;
        }

        // Handle transition from grounded to not grounded
        if (wasGrounded && !IsGrounded && !hasJumped)
        {
            coyoteTimer = coyoteTimeSeconds;
        }

        if (IsGrounded)
        {
            Position = Position with {Y = groundHit.position.Y + playerHeight};
            physicsVelocity.Y = Math.Max(0, physicsVelocity.Y);

            // Handle jump buffer when landing
            if (jumpBufferTimer > 0 && !isCrouching)
            {
                physicsVelocity.Y = jumpImpulse;
                hasJumped = true;
                jumpBufferTimer = 0;
            }
            else
            {
                hasJumped = false;
            }
        }

        // Update camera's look controls
        Camera.HandleLook(deltaTime);

        // Update crouch state
        isCrouching = Keyboard.IsKeyPressed(Key.ShiftLeft);

        // Handle movement input relative to camera orientation
        var forward = Vector3.Normalize(Camera.Forward with { Y = 0 });
        var right = Vector3.Normalize(Camera.Right with { Y = 0 });

        Vector3 moveForce = Vector3.Zero;
        
        bool isRunning = Keyboard.IsKeyPressed(Key.ControlLeft) && !isCrouching;
        
        float currentAcceleration = isRunning ? runAcceleration : acceleration;
        float currentMaxSpeed = isRunning ? runSpeed : walkSpeed;

        if (isCrouching)
        {
            currentAcceleration *= crouchSpeedMultiplier;
            currentMaxSpeed *= crouchSpeedMultiplier;
        }

        if (Keyboard.IsKeyPressed(Key.W))
            moveForce += forward * currentAcceleration;
        if (Keyboard.IsKeyPressed(Key.S))
            moveForce -= forward * currentAcceleration;
        if (Keyboard.IsKeyPressed(Key.D))
            moveForce += right * currentAcceleration;
        if (Keyboard.IsKeyPressed(Key.A))
            moveForce -= right * currentAcceleration;

        // Only apply movement force if not facing a too-high step or moving away from it
        if (!isStepTooHigh || Vector3.Dot(moveForce, lastMoveDirection) < 0)
        {
            movementVelocity += moveForce * dt;
        }
        else
        {
            // If facing a too-high step, only allow movement perpendicular to the step
            Vector3 stepNormal = Vector3.Normalize(lastMoveDirection);
            Vector3 perpendicular = new Vector3(-stepNormal.Z, 0, stepNormal.X);
            float perpendicularComponent = Vector3.Dot(moveForce, perpendicular);
            movementVelocity += perpendicular * perpendicularComponent * dt;
        }

        if (IsGrounded && moveForce.Length() == 0)
        {
            var decel = deceleration * dt;
            if (movementVelocity.Length() < decel)
            {
                movementVelocity = Vector3.Zero;
            }
            else
            {
                movementVelocity -= Vector3.Normalize(movementVelocity) * decel;
            }
        }

        // Clamp movement speed using current max speed
        if (movementVelocity.Length() > currentMaxSpeed)
        {
            movementVelocity = Vector3.Normalize(movementVelocity) * currentMaxSpeed;
        }

        // Apply friction to physics velocity when grounded
        if (IsGrounded)
        {
            var friction = 0.1f;
            var horizontalPhysicsVel = physicsVelocity with { Y = 0 };
            if (horizontalPhysicsVel.Length() > 0)
            {
                var frictionDir = Vector3.Normalize(horizontalPhysicsVel);
                physicsVelocity -= frictionDir * Math.Min(friction, horizontalPhysicsVel.Length());
            }
        }

        // Handle height interpolation and visual position updates
        currentHeightMultiplier = EngineMath.Lerp(
            currentHeightMultiplier,
            isCrouching ? crouchHeightMultiplier : 1f,
            heightLerpSpeed * dt
        );

        if (IsGrounded)
        {
            float targetHeight = Position.Y - (playerHeight * (1f - currentHeightMultiplier));
            VisualPosition = Position with { Y = EngineMath.Lerp(VisualPosition.Y, targetHeight, heightLerpSpeed * dt) };
        }
        else
        {
            VisualPosition = Position with { Y = Position.Y - (playerHeight * (1f - currentHeightMultiplier)) };
        }

        Camera.Position = VisualPosition + new Vector3(0, playerHeight * currentHeightMultiplier, 0);

        // Apply final velocity, taking into account step blocking
        var combinedVelocity = physicsVelocity + movementVelocity;
        
        // If facing a too-high step, remove the velocity component in the direction of the step
        if (isStepTooHigh)
        {
            Vector3 horizontalVelocity = combinedVelocity with { Y = 0 };
            if (Vector3.Dot(horizontalVelocity, lastMoveDirection) > 0)
            {
                // Project velocity onto the plane perpendicular to lastMoveDirection
                Vector3 perpendicular = new Vector3(-lastMoveDirection.Z, 0, lastMoveDirection.X);
                float perpendicularComponent = Vector3.Dot(horizontalVelocity, perpendicular);
                horizontalVelocity = perpendicular * perpendicularComponent;
                combinedVelocity = horizontalVelocity + new Vector3(0, combinedVelocity.Y, 0);
            }
        }

        Position += combinedVelocity * dt;

        if (combinedVelocity.Length() > 0)
        {
            var horizontalVelocity = combinedVelocity with { Y = 0 };
            if (horizontalVelocity.Length() > 0)
            {
                lastMoveDirection = Vector3.Normalize(horizontalVelocity);
            }
        }
        
        wasGrounded = IsGrounded;
    }
}