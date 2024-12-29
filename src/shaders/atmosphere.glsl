const vec3 sunDir = vec3(0.5, 0.3, 0.5);

// Hash function for stable random values (stars)
float hash(vec3 p) {
    p = fract(p * vec3(443.8975,397.2973, 491.1871));
    p += dot(p.xyz, p.yzx + 19.19);
    return fract(p.x * p.y * p.z);
}

// Convert direction to stable star coordinates
vec3 getStableStarCoord(vec3 dir) {
    float cells = 1024.0;
    vec3 stableDir = floor(dir * cells) / cells;
    return normalize(stableDir);
}

// Calculate sun color based on its height
vec3 calculateSunColor(vec3 sunDir) {
    float sunHeight = sunDir.y;
    
    // Base sun color (pure white)
    vec3 baseSunColor = vec3(1.0, 1.0, 1.0);
    
    // Atmospheric thickness based on sun height
    float atmosphereThickness = 1.0 / (max(sunHeight, 0.0) + 0.1);
    
    // Rayleigh scattering coefficients for sunlight
    vec3 rayleighSun = vec3(5.5, 13.0, 22.4) * 0.005;
    
    // Calculate extinction of different wavelengths
    vec3 extinction = exp(-rayleighSun * atmosphereThickness);
    
    // Sunset/sunrise colors
    vec3 sunsetColor = vec3(1.0, 0.6, 0.3);
    vec3 dayColor = vec3(1.0, 0.98, 0.95);
    
    // Interpolate between sunset and day colors based on height
    float sunsetFactor = smoothstep(0.15, 0.25, sunHeight);
    vec3 finalSunColor = mix(sunsetColor, dayColor, sunsetFactor);
    
    // Apply extinction
    finalSunColor *= extinction;
    
    // Fade out sun color during night
    float daylight = smoothstep(-0.1, 0.1, sunHeight);
    return finalSunColor * daylight;
}

// Calculate star visibility
float calculateStars(vec3 dir) {
    vec3 stableDir = getStableStarCoord(dir);
    float star = (hash(stableDir * 1000.0) < 0.0003) ? 1.0 : 0.0;
    float brightness = hash(stableDir * 789.0);
    star *= smoothstep(0.95, 1.0, brightness);
    float twinkle = hash(stableDir + fract(time * 0.1));
    star *= 0.8 + 0.2 * twinkle;
    return star;
}

// Atmospheric scattering approximation
vec3 atmosphere(vec3 dir, vec3 sunDir) {
    float sunHeight = sunDir.y;
    float height = dir.y;
    
    // Calculate sun color
    vec3 currentSunColor = calculateSunColor(sunDir);
    
    // Rayleigh scattering
    float rayleighStrength = exp(-0.00287 + height * 0.459);
    
    // Mie scattering
    float sunDot = max(0.0, dot(dir, sunDir));
    float miePhase = 1.5 * ((1.0 - sunDot * sunDot) / (2.0 + sunDot * sunDot));
    
    // Atmosphere density
    float density = exp(-(height * 4.0)) * (1.0 + height * 0.5);
    
    // Sky colors based on sun height
    vec3 dayColor = vec3(0.2, 0.4, 1.0);
    vec3 sunsetColor = vec3(0.8, 0.6, 0.5);
    float sunsetFactor = smoothstep(0.0, 0.4, sunHeight);
    vec3 skyColor = mix(sunsetColor, dayColor, sunsetFactor);
    
    // Combine scattering
    vec3 sky = skyColor * rayleighStrength +
               currentSunColor * miePhase * density;
               
    // Sun disk with new coloring
    float sunDisk = smoothstep(0.999, 0.9999, sunDot);
    float sunGlow = pow(sunDot, 4.0) * 0.2;
    
    // Night sky
    float nightness = smoothstep(0.1, -0.1, sunHeight);
    
    // Stars
    float star = calculateStars(dir);
    vec3 stars = vec3(star) * nightness;
    
    // Day/night transition
    float dayFactor = smoothstep(-0.1, 0.1, sunHeight);
    sky = mix(sky * 0.02, sky, dayFactor); // More dramatic night dimming
    
    // Final color combination
    vec3 finalColor = sky + 
                     sunDisk * currentSunColor * 50.0 +
                     sunGlow * currentSunColor +
                     stars;
    
    // Horizon effects
    float horizonFade = smoothstep(-0.1, 0.2, height);
    finalColor = mix(
        mix(vec3(0.7, 0.75, 0.8) * dayFactor, finalColor, horizonFade),
        finalColor,
        smoothstep(0.0, 0.2, abs(height))
    );
    
    return finalColor;
}

vec3 getSkyColor(vec3 dir) {
    return atmosphere(normalize(dir), normalize(sunDir));
}