<?php

namespace Database\Seeders;

// use Illuminate\Database\Console\Seeds\WithoutModelEvents;

use App\Models\orderInvoice;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // \App\Models\User::factory(10)->create();

        // \App\Models\User::factory()->create([
        //     'name' => 'Test User',
        //     'email' => 'test@example.com',
        // ]);

        $this->call(AdminSeeder::class);
        $this->call(EmployeeSeeder::class);
        $this->call(UserSeeder::class);
        $this->call(TableSeeder::class);
        $this->call(ReservationSeeder::class);
        $this->call(FoodSeeder::class);
        $this->call(CategorySeeder::class);
        $this->call(FoodCategorySeeder::class);
        $this->call(OrderInvoiceSeeder::class);
        $this->call(OrderInvoiceDetailsSeeder::class);
    }
}
