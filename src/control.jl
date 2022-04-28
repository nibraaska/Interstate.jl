using Statistics
using DataStructures: CircularDeque
using Flux, Flux.Optimise
using Flux: onehotbatch, onecold, crossentropy, Momentum, params, ADAM
using Base.Iterators: partition
using NNlib
using Serialization
using StatsBase: sample
using BSON: @save, @load
using Plots;

function wrap(θ)
    mod(θ += pi, 2*pi) - pi
end

function clip(x, l)
    max(-l, min(l, x))
end

MAX_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 100
DISCOUNT = 0.9
UPDATE_TARGET_EVERY = 10
EXPLORE = 0.05
opt = ADAM()

mutable struct Memory
    sₜ
    aₜ
    rₙ
    sₙ
    done
end

function model()
    if isfile("mymodel.bson")
        @load "mymodel.bson" target_model
        return target_model
    else
        m = Chain(
            Dense(550, 1024, σ),
            Dense(1024, 64, σ),
            Dense(64, 4),
            softmax
        )
        return m
    end
end

function update_weights!(target_model, main_model)
    for (main_param, target_param) in zip(params(main_model), params(target_model))
        target_param .= main_param
    end
end

function update_memory!(mem, replay_mem)
    push!(replay_mem, mem)
    # if length(replay_mem) < MAX_REPLAY_MEMORY_SIZE 
    #     push!(replay_mem, mem)
    # else
    #     popfirst!(replay_mem)
    #     push!(replay_mem, mem)
    # end
end

function train!(replay_memory, main_model, target_model, terminal_state, target_update_counter)
    
    loss(x, y) = Flux.Losses.mse(main_model(x), y)

    minibatch = sample(replay_memory, MINIBATCH_SIZE, replace=false)

    if terminal_state
        minibatch[MINIBATCH_SIZE] = replay_memory[length(replay_memory)]
    end

    current_states = hcat((transition.sₜ for transition in minibatch)...)
    current_qs_list = main_model(current_states)
    
    new_current_states = hcat((transition.sₙ for transition in minibatch)...)
    future_qs_list = target_model(new_current_states)
    X = []
    y = []

    for (i, data_point) in enumerate(minibatch)
        sₜ, aₜ, rₙ, sₙ, done = data_point.sₜ, data_point.aₜ, data_point.rₙ, data_point.sₙ, data_point.done
        if done
            new_q = rₙ
        else
            new_q = rₙ + DISCOUNT * maximum(future_qs_list[:,i])
        end
        current_qs = current_qs_list[:,i]
        current_qs[aₜ,:] .= new_q
        push!(X, sₜ)
        push!(y, current_qs)
    end
    
    X = hcat((X)...);
    y = hcat((y)...);
    
    ps = Flux.params(main_model);
    data = Flux.Data.DataLoader((X, y),shuffle=true);
    Flux.train!(loss, ps, data, opt)
    
    # if target_update_counter > UPDATE_TARGET_EVERY || terminal_state
    #     println("In here")
    #     update_weights!(target_model, main_model)
    #     target_update_counter = 0
    # end

end

function keyboard_controller(KEY::Channel, 
                             CMD::Channel, 
                             SENSE::Channel, 
                             EMG::Channel;
                             K1=5, 
                             K2=.5, 
                             disp=false, 
                             V=0.0, 
                             θ = 0.0, 
                             V_max = 100.0, 
                             θ_step=0.1, 
                             V_step = 1.5)
    println("Keyboard controller in use.")
    println("Press 'i' to speed up,")
    println("Press 'k' to slow down,")
    println("Press 'j' to turn left,")
    println("Press 'l' to turn right.")

    while true
        sleep(0)
        @return_if_told(EMG)
        
        key = @take_or_default(KEY, ' ')
        meas = @fetch_or_continue(SENSE)

        speed = meas.speed
        heading = meas.heading
        segment = meas.road_segment_id
        if key == 'i'
            V = min(V_max, V+V_step)
        elseif key == 'j' 
            θ += θ_step
        elseif key == 'k'
            V = max(0, V-V_step)
        elseif key == 'l'
            θ -= θ_step
        end
        err_1 = V-speed
        err_2 = clip(θ-heading, π/2)
        cmd = [K1*err_1, K2*err_2]
        @replace(CMD, cmd)
        if disp
            print("\e[2K")
            print("\e[1G")
            @printf("Command: %f %f, speed: %f, segment: %d", cmd..., speed, segment)
        end
    end
end

function controller(CMD::Channel, 
                    SENSE::Channel, 
                    SENSE_FLEET::Channel, 
                    EMG::Channel,
                    road;
                    V=0.0, 
                    θ = 0.0, 
                    V_max = 100.0, 
                    θ_step=0.1, 
                    V_step = 1.5,
                    iteration=1)
    ego_meas = fetch(SENSE)
    fleet_meas = fetch(SENSE_FLEET)
    K₁ = K₂ = 0.5
    e = 0
    target_update_counter = 0  
    main_model = model()
    target_model = model() 
    
    replay_memory = Memory[]
    while true
        sleep(0)
        @return_if_told(EMG)
        
        ego_meas = @fetch_or_default(SENSE, ego_meas)
        fleet_meas = @fetch_or_default(SENSE_FLEET, fleet_meas)

        ego_meas_list = [ego_meas.position[1] ego_meas.position[2] ego_meas.speed ego_meas.heading ego_meas.road_segment_id ego_meas.target_lane ego_meas.target_vel ego_meas.front ego_meas.rear ego_meas.left ego_meas.right]
        fleat_meas_tmp = [ [fleet_meas[key].position[1] fleet_meas[key].position[2] fleet_meas[key].speed fleet_meas[key].heading fleet_meas[key].road_segment_id fleet_meas[key].target_lane fleet_meas[key].target_vel fleet_meas[key].front fleet_meas[key].rear fleet_meas[key].left fleet_meas[key].right ] for key in keys(fleet_meas)]
        fleat_meas_list = hcat((fleat_meas_tmp)...);

        input_meas = transpose([ego_meas_list fleat_meas_list])
        action = argmax(target_model(input_meas))[1]
        speed = ego_meas.speed
        heading = ego_meas.heading
        segment = ego_meas.road_segment_id
        if rand(1)[1] < EXPLORE
            action = rand(1:4)[1]
        end
        if action == 1
            V = min(V_max, V+V_step)
        elseif action == 2
            θ += θ_step
        elseif action == 3
            V = max(0, V-V_step)
        elseif action == 4
            θ -= θ_step
        else
            println("ERROR")
        end
        # println(action)
        err_1 = V-speed
        err_2 = clip(θ-heading, π/2)
        command = [K₁*err_1, K₂*err_2]
        
        terminal = false
        e = @fetch_or_default(EMG, e)
        if e == 1
            terminal = true
            mem_past = Memory(input_meas, action, -100, input_meas, true)
            update_memory!(mem_past, replay_memory)
            # println("Final length", length(replay_memory))
            if length(replay_memory) >= MINIBATCH_SIZE
                for i ∈ 1:10
                    for j ∈ 1:10
                        train!(replay_memory, main_model, target_model, terminal, target_update_counter) 
                    end
                    update_weights!(target_model, main_model)
                end
                if isfile("mymodel.bson")
                    rm("mymodel.bson")
                end
                serialize("data/replay_memory_$iteration.dat", replay_memory)
                @save "mymodel.bson" target_model
            end
            # println(target_update_counter)
            # update_memory!(mem_past, replay_memory)
            # if len >= MINIBATCH_SIZE
            #     train!(replay_memory, main_model, target_model, terminal, target_update_counter) 
            # end
            # @save "mymodel.bson" target_model
        else
            mem_past = Memory(input_meas, action, ego_meas.time, input_meas, false)
            len = length(replay_memory)
            if len > 0
                replay_memory[len].sₙ = input_meas
            end
            update_memory!(mem_past, replay_memory)
            # len = length(replay_memory)
            # if len > 0
            #     replay_memory[len].sₙ = input_meas
            # end
            # if len >= MINIBATCH_SIZE
            #     train!(replay_memory, main_model, target_model, terminal, target_update_counter) 
            # end
            # update_memory!(mem_past, replay_memory)
        end
        target_update_counter = target_update_counter + 1
        # println(target_update_counter)
        @replace(CMD, command)
    end
end
