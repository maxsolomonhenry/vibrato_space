function out_ = hz_to_midi(hz, EPS)
% Convert to from frequency to midi (pitch in logspace).
    
    if nargin == 1
        EPS = 1e-8; 
    end
    
    out_ = 12 * log2((hz + EPS)/440) + 69;
end