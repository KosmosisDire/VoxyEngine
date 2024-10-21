#version 460 core

in vec2 TexCoords;            // Texture coordinates
out float FragColor;          // Final SSAO result

// Uniform inputs
uniform sampler2D positionTexture;  // Position texture
uniform sampler2D normalTexture;    // Normal texture
uniform sampler2D noiseTexture;     // Noise texture
uniform vec3 samples[64];           // SSAO kernel samples
uniform mat4 projection;            // Projection matrix
uniform vec2 resolution;            // Screen resolution

const int kernelSize = 64;          // Size of the SSAO kernel
const float radius = 0.5;           // Sampling radius
const float bias = 0.1;           // Bias to prevent self-shadowing



void main()
{

    vec2 noiseScale = vec2(resolution.x / 4.0, resolution.y / 4.0); // scale noise (smaller for higher resolution)

    // get input for SSAO algorithm
    vec3 fragPos = texture(positionTexture, TexCoords).xyz;
    vec3 normal = normalize(texture(normalTexture, TexCoords).rgb);
    vec3 randomVec = normalize(texture(noiseTexture, TexCoords * noiseScale).xyz);

    // create TBN change-of-basis matrix: from tangent-space to view-space
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    // iterate over the sample kernel and calculate occlusion factor
    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; ++i)
    {
        // get sample position
        vec3 samplePos = TBN * samples[i]; // from tangent to view-space
        samplePos = fragPos + samplePos * radius; 
        
        // project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0
        
        // get sample depth
        float sampleDepth = texture(positionTexture, offset.xy).z; // get depth value of kernel sample
        
        // range check & accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;           
    }
    occlusion = 1.0 - (occlusion / kernelSize);
    
    FragColor = occlusion; // output occlusion factor
}
