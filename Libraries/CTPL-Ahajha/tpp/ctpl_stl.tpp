/*********************************************************
*
*  Copyright (C) 2014 by Vitaliy Vitsentiy
*  Modifications: Copyright (C) 2021 by Alex Trotta
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*********************************************************/

template <typename T>
bool detail::atomic_queue<T>::push(T&& value)
{
	std::lock_guard lock(this->mut);
	this->q.push(std::forward<T>(value));
	return true;
}

template <typename T>
bool detail::atomic_queue<T>::pop(T & value)
{
	std::lock_guard lock(this->mut);
	if (this->q.empty())
		return false;
	value = std::move(this->q.front());
	this->q.pop();
	return true;
}

template <typename T>
bool detail::atomic_queue<T>::empty() const
{
	std::lock_guard lock(this->mut);
	return this->q.empty();
}

template <typename T>
void detail::atomic_queue<T>::clear()
{
	std::lock_guard lock(this->mut);
	while (!this->q.empty()) this->q.pop();
}

thread_pool::thread_pool(std::size_t n_threads) :
	done(false), stopped(false), _n_idle(0)
{
	// Starts with 0 threads, resize.
	this->resize(n_threads);
}

thread_pool::~thread_pool()
{
	this->stop(true);
}

std::size_t thread_pool::size() const
{
	return this->threads.size();
}

std::size_t thread_pool::n_idle() const
{
	return this->_n_idle;
}

void thread_pool::resize(std::size_t n_threads)
{
	// One of these will be true if this->stop() has been called,
	// resizing in that case would be pointless at best or incorrect at worst.
	if (this->stopped || this->done) return;
	
	const std::size_t old_n_threads = this->threads.size();
	
	// If the number of threads stays the same or increases
	if (old_n_threads <= n_threads)
	{
		// Start up all threads into their main loops.
		while (this->threads.size() < n_threads)
		{
			this->emplace_thread();
		}
	}
	else // the number of threads is decreased
	{
		// For each thread to be removed
		for (std::size_t i = n_threads; i < old_n_threads; ++i)
		{
			// Tell the thread to finish its current task
			// (if it has one) and stop. Detach the thread, since
			// there is no need to wait for it to finish (join).
			*this->stop_flags[i] = true;
			this->threads[i].detach();
		}
		{
			// Stop any detached threads that were waiting.
			// All other detached threads will eventually stop.
			std::lock_guard lock(this->signal_mut);
			this->signal.notify_all();
		}
		
		// Safe to delete because the threads are detached.
		this->threads.resize(n_threads);
		
		// Safe to delete because the threads have copies of
		// the shared pointers to their respective flags, not originals.
		this->stop_flags.resize(n_threads);
	}
}

void thread_pool::clear_queue()
{
	this->tasks.clear();
}

void thread_pool::wait()
{
	std::unique_lock lock(this->waiter_mut);
	waiter.wait(lock);
}

void thread_pool::stop(bool finish)
{
	// Force the threads to stop
	if (!finish)
	{
		// If this->stop(false) has already been called, no need to stop again.
		// If this->stop(true) has alredy been called, still continue, as this
		// will stop the completion of the rest of the tasks in the queue.
		if (this->stopped) return;
		
		this->stopped = true;
		
		// Command all threads to stop
		for (auto& stop_flag : this->stop_flags)
		{
			*stop_flag = true;
		}
		
		// Remove any remaining tasks.
		this->clear_queue();
	}
	else // Let the threads continue
	{
		// If this->stop() has been already been called, no need to stop again.
		if (this->done || this->stopped) return;
		
		// Give the waiting threads a command to finish
		this->done = true;
	}
	{
		// Stop all waiting threads
		std::lock_guard lock(this->signal_mut);
		this->signal.notify_all();
	}
	
	// Wait for the computing threads to finish
	for (auto& thr : this->threads)
	{
		if (thr.joinable())
			thr.join();
	}
	
	// Release all resources.
	this->clear_queue();
	this->threads.clear();
	this->stop_flags.clear();
}

template<typename F, typename... Args>
std::future<std::invoke_result_t<F,Args...>> thread_pool::push(F && f, Args&&... args)
{
	// std::packaged_task is used to get a future out of the function call.
	auto pck = std::make_shared<std::packaged_task<std::invoke_result_t<F,Args...>()>>(
		// This has been tested to ensure perfect forwarding still occurs with
		// the parameters captured by reference.
		[&f,&args...]
		{
			if constexpr (sizeof...(args) == 0)
			{
				// Only need to forward the function.
				return std::forward<F>(f);
			}
			else
			{
				// std::forward is used to ensure perfect
				//     forwarding of rvalues where necessary.
				// std::bind is used to make a parameterless function
				//     that simulates calling f with its respective arguments.
				return std::bind(std::forward<F>(f), std::forward<Args>(args)...);
			}
		}()
	);
	
	// Lastly, create a function that wraps the packaged
	// task into a signature of void().
	this->tasks.push(std::make_unique<std::function<void()>>(
		[pck]{ (*pck)(); })
	);
	
	// Notify one waiting thread so it can wake up and take this new task.
	std::lock_guard lock(this->signal_mut);
	this->signal.notify_one();
	
	// Return the future, the user is now responsible
	// for the return value of the task.
	return pck->get_future();
}

void thread_pool::emplace_thread()
{
	this->stop_flags.emplace_back(std::make_shared<std::atomic<bool>>(false));
	
	// The main loop for the thread. Grabs a copy of the pointer
	// to the stop flag.
	this->threads.emplace_back([this, stop_flag = this->stop_flags.back()]
	{
		std::atomic<bool> & stop = *stop_flag;
		
		// Used to store new tasks.
		std::unique_ptr<std::function<void()>> task;
		
		// True if 'task' currently has a runnable task in it.
		bool has_new_task = this->tasks.pop(task);
		while (true)
		{
			// If there is a task to run
			while (has_new_task)
			{
				// Run the task
				(*task)();
				
				// Delete the task
				task.reset();
				
				// The thread is wanted to stop, return even
				// if the queue is not empty yet
				if (stop) return;
				
				// Get a new task
				has_new_task = this->tasks.pop(task);
			}
			
			// At this point the queue has run out of tasks, wait here for more.
			
			// Thread is now idle.
			// If all threads are idle, notify any waiting in wait().
			if (++this->_n_idle == this->size())
			{
				std::lock_guard lock(this->waiter_mut);
				this->waiter.notify_all();
			}
			
			std::unique_lock lock(this->signal_mut);
			
			// While the following evaluates to true, wait for a signal.
			this->signal.wait(lock, [this, &task, &has_new_task, &stop]()
			{
				// Try to get a new task. This will fail if the thread was
				// woken up for another reason (stopping or resizing), or
				// if another thread happened to grab the task before this
				// one got to it.
				has_new_task = this->tasks.pop(task);
				
				// If there is a new task or the thread is being told to stop,
				// stop waiting.
				return has_new_task || this->done || stop;
			});
			
			// Thread is no longer idle.
			--this->_n_idle;
			
			// if the queue is empty and it was able to stop waiting, then
			// that means the thread was told to stop, so stop.
			if (!has_new_task) return;
		}
	});
}
